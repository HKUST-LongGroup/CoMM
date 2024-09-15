import os
import random
from typing import Any, Optional, Dict, List

import torch
from lightning.pytorch import LightningModule
from transformers import get_linear_schedule_with_warmup, CLIPTextModel, CLIPTokenizer, PreTrainedTokenizer
from torch.optim import AdamW
# from deepspeed.ops.adam import FusedAdam
import torch.nn as nn

from minigpt4.models.mini_gpt5 import MiniGPT5
from minigpt4.common.config import Config

from diffusers import AutoencoderKL, UNet2DConditionModel
# import wandb
import torch.nn.functional as F

from constants import *
from diffusers import StableDiffusionPipeline
from diffusers.models.vae import DiagonalGaussianDistribution


class MiniGPT5_InputProcessor(object):

    def __init__(self, tokenizer: PreTrainedTokenizer, image_processor: Any):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, text=None, images=None, **kwargs) -> Any:
        output_dict = {}
        if text is not None:
            text_output = self.tokenizer(text,
                                         return_tensors="pt",
                                         padding=True,
                                         max_length=self.tokenizer.model_max_length,
                                         truncation=True,
                                         **kwargs)
            output_dict.update(text_output)

        if images is not None:
            all_images = []
            if isinstance(images, list):
                for img in images:
                    image_output = self.image_processor(img)
                    all_images.append(image_output)
                input_images = torch.stack(all_images, dim=0)
            else:
                input_images = self.image_processor(images)
            output_dict['input_images'] = input_images
        return output_dict


class MiniGPT4Args:
    cfg_path = "config/minigpt4.yaml"
    options = []


# define the LightningModule
class MiniGPT5_Model(LightningModule):
    def __init__(self,
        encoder_model_config,
        device,
        device2,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=['encoder_model_config'])
        self.encoder_model_config = encoder_model_config
        self.input_vis_processor = None
        self.device2 = device2

        if encoder_model_config.model_type == 'multimodal_encoder':
            minigpt4_config = Config(MiniGPT4Args)
            self.model = MiniGPT5.from_config(minigpt4_config.model_cfg)
            self.tokenizer = self.model.llama_tokenizer

            hidden_size = self.model.llama_model.config.hidden_size
        elif encoder_model_config.model_type == 'llama2':
            MiniGPT4Args.cfg_path = "config/minigpt4_llama2.yaml"
            minigpt4_config = Config(MiniGPT4Args)
            self.model = MiniGPT5.from_config(minigpt4_config.model_cfg)
            self.tokenizer = self.model.llama_tokenizer

            hidden_size = self.model.llama_model.config.hidden_size
        else:
            raise NotImplementedError

        sd_model_name = "stabilityai/stable-diffusion-2-1-base"

        self.sd_text_encoder = CLIPTextModel.from_pretrained(sd_model_name, subfolder="text_encoder").to(PRECISION).to(
            device2)
        self.sd_tokenizer = CLIPTokenizer.from_pretrained(sd_model_name, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae").to(PRECISION)

        self.unet = UNet2DConditionModel.from_pretrained(sd_model_name, subfolder="unet").to(PRECISION)
        self.unet.enable_gradient_checkpointing()
        self.vae.requires_grad_(False)
        self.sd_text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        sd_hidden_size = self.sd_text_encoder.config.hidden_size
        self.t2i_decoder_prompt = torch.nn.Parameter(
            torch.randn((1, 77, sd_hidden_size), dtype=TRAINABLE_PRECISION).to(device2))
        self.llm_to_t2i_mapping = nn.Transformer(batch_first=True, norm_first=True, d_model=sd_hidden_size,
                                                 num_encoder_layers=4, num_decoder_layers=4,
                                                 dim_feedforward=sd_hidden_size * 4, dropout=0.0,
                                                 dtype=TRAINABLE_PRECISION)

        if len(ALL_IMG_TOKENS):
            self.output_img_id = self.tokenizer.convert_tokens_to_ids(ALL_IMG_TOKENS[0])
        self.img_token_num = IMG_TOKEN_NUM

        self.image_pipeline = StableDiffusionPipeline.from_pretrained(
            sd_model_name,
            vae=self.vae,
            unet=self.unet,
            safety_checker=None,
        ).to(PRECISION)

        self.noise_scheduler = self.image_pipeline.scheduler

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, sd_hidden_size),
            nn.GELU(),
            nn.Linear(sd_hidden_size, sd_hidden_size),
        ).to(TRAINABLE_PRECISION)

        empty_text_feature = self.encode_caption('', self.sd_tokenizer.model_max_length, inference=True)
        self.register_buffer('empty_text_feature', empty_text_feature, persistent=False)

        zero_img_feature = torch.zeros((1, self.img_token_num, hidden_size), dtype=TRAINABLE_PRECISION).cuda()
        self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)

    def training_step(self, batch, batch_idx):
        for key in batch.keys():
            if type(batch[key]) == list:
                batch[key] = batch[key]
            else:
                batch[key] = batch[key].to(self.device)

        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        source_text = batch['source']
        target_text = batch['target']
        captions = batch['caption']
        input_images = batch.get('input_images', None)
        output_image = batch.get('output_image', None)
        input_images_feature = batch.get('input_images_feature', None)
        output_image_feature = batch.get('output_image_feature', None)

        bs = len(source_text)
        loss_dict = self(input_ids, attention_mask, input_images, output_image, labels, captions, input_images_feature,
                         output_image_feature)
        loss = loss_dict['loss']
        log_dict = {f'train_{k}': v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        for key in batch.keys():
            if type(batch[key]) == list:
                batch[key] = batch[key]
            else:
                batch[key] = batch[key].to(self.device)

        input_ids = batch['input_ids']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        source_text = batch['source']
        target_text = batch['target']
        captions = batch['caption']
        input_images = batch.get('input_images', None)
        output_image = batch.get('output_image', None)
        input_images_feature = batch.get('input_images_feature', None)
        output_image_feature = batch.get('output_image_feature', None)

        bs = len(source_text)
        loss_dict = self(input_ids, attention_mask, input_images, output_image, labels, captions, input_images_feature,
                         output_image_feature)
        log_dict = {f'val_{k}': v for k, v in loss_dict.items()}
        self.log_dict(log_dict, batch_size=bs, logger=True, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self, is_deep_speed=True):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if p.requires_grad],
            },
            {
                "params": [p for n, p in self.fc.named_parameters() if p.requires_grad],
                "lr": self.hparams.learning_rate * 10,
            },
            {
                "params": [p for n, p in self.llm_to_t2i_mapping.named_parameters() if p.requires_grad],
                "lr": self.hparams.learning_rate * 10,
            },
            {
                "params": self.t2i_decoder_prompt,
                "lr": self.hparams.learning_rate * 10,
            },
            {
                "params": [p for n, p in self.unet.named_parameters() if p.requires_grad],
                "lr": self.hparams.learning_rate * 10,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                            eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=1000
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, input_ids, attention_mask, input_images, output_image, labels, captions=None,
                input_images_feature=None, output_image_feature=None, image_loss_only=False):
        outputs, special_token_index = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask,
                                                  input_images=input_images,
                                                  input_img_features=input_images_feature,
                                                  output_hidden_states=True)
        text_loss = outputs['loss']
        last_hidden_state = outputs['hidden_states'][-1]
        t2i_input_embedding = []
        caption_feature = []
        calculate_caption_loss = False
        for i in range(len(special_token_index)):
            bs_id, seq_id = special_token_index[i]
            # random set 10% data with empty text feature
            if USE_CFG and random.random() < 0.1:
                t2i_input_embedding.append(self.zero_img_feature)
                if calculate_caption_loss:
                    caption_feature.append(self.empty_text_feature)
            else:
                t2i_input_embedding.append(last_hidden_state[bs_id:bs_id + 1, seq_id:seq_id + self.img_token_num, :])
                if calculate_caption_loss:
                    caption_feature.append(
                        self.encode_caption(captions[bs_id][i], self.sd_tokenizer.model_max_length, inference=True))

        if len(t2i_input_embedding) == 0:
            loss = 0.01 * text_loss
            if calculate_caption_loss:
                return {'loss': loss, 'text_loss': text_loss, 'image_loss': 0.0, 'caption_loss': 0.0}
            else:
                return {'loss': loss, 'text_loss': text_loss, 'image_loss': 0.0}

        else:
            t2i_input_embedding = torch.cat(t2i_input_embedding, dim=0)
            img_token_bs = t2i_input_embedding.shape[0]
            t2i_input_embedding = self.fc(t2i_input_embedding)
            
            if self.device2 is not None:
                t2i_input_embedding = t2i_input_embedding.to(self.device2)
                output_image = output_image.to(self.device2)
                text_loss = text_loss.to(self.device2)

            mapping_feature = self.llm_to_t2i_mapping(src=t2i_input_embedding,
                                                      tgt=self.t2i_decoder_prompt.repeat(img_token_bs, 1, 1))

            if output_image_feature is None:
                # image_loss = self.compute_vae_loss(mapping_feature, output_image)
                image_loss = self.compute_image_loss(mapping_feature, output_image)
            else:
                raise NotImplementedError

            if calculate_caption_loss:
                caption_feature = torch.cat(caption_feature, dim=0)
                caption_loss = F.mse_loss(mapping_feature, caption_feature)
                if image_loss_only:
                    loss = image_loss + 0.1 * caption_loss
                else:
                    loss = 0.01 * text_loss + image_loss + 0.1 * caption_loss

                return {'loss': loss, 'text_loss': text_loss, 'image_loss': image_loss, 'caption_loss': caption_loss}
            else:
                if image_loss_only:
                    loss = image_loss
                else:
                    loss = 0.01 * text_loss + image_loss

                return {'loss': loss, 'text_loss': text_loss, 'image_loss': image_loss}

    def generate(self, utterance, input_image=None, task_name=None, max_new_tokens=1024, force_generation=False,
                 guidance_scale=7.5) -> Any:
        if input_image is None:
            input_image = torch.zeros((1, 3, 224, 224), dtype=PRECISION).to(self.device)

        if type(utterance) == str:
            utterance = [utterance]
        llm_sample_outputs = self.model.predict(utterance, input_image, max_new_tokens=max_new_tokens, temperature=1.0,
                                                repetition_penalty=1.0, task_name=task_name,
                                                force_generation=force_generation)
        new_tokens = llm_sample_outputs['sequences'][0]
        pred_out = self.tokenizer.decode(new_tokens)
        # print(f'Generated text: {pred_out}')

        last_hidden_state = llm_sample_outputs['hidden_states']
        special_token_index = (new_tokens == self.output_img_id).nonzero()
        predicted_images_ft = None if len(special_token_index) == 0 else []

        t2i_input_embedding_list = []
        for i in range(len(special_token_index)):
            idx = special_token_index[i, 0]
            try:
                t2i_input_embedding = last_hidden_state[idx][-1]
                assert t2i_input_embedding.shape[1] == self.img_token_num
                img0_output_feature = last_hidden_state[idx - 1][-1][:, -1:]
                t2i_input_embedding = torch.cat([img0_output_feature, t2i_input_embedding[:, :-1]], dim=1)
                t2i_input_embedding = self.fc(t2i_input_embedding)
                t2i_input_embedding_list.append(t2i_input_embedding)
            except:
                continue
        if len(t2i_input_embedding_list) == 0:
            return pred_out, []
        t2i_input_embedding_list = torch.cat(t2i_input_embedding_list, dim=0)
        t2i_input_embedding_list = t2i_input_embedding_list.to(self.device2)
        for i in range(len(special_token_index)):
            t2i_input_embedding = t2i_input_embedding_list[i]
            t2i_input_embedding = t2i_input_embedding.unsqueeze(0)
            mapping_feature = self.llm_to_t2i_mapping(src=t2i_input_embedding, tgt=self.t2i_decoder_prompt)
            if USE_CFG:
                empty_feature = self.fc(self.zero_img_feature)
                empty_feature = self.llm_to_t2i_mapping(src=empty_feature, tgt=self.t2i_decoder_prompt)
                cur_predicted_images_ft = \
                    self.image_pipeline(prompt_embeds=mapping_feature, negative_prompt_embeds=empty_feature,
                                        guidance_scale=guidance_scale).images[0]
            else:
                cur_predicted_images_ft = \
                    self.image_pipeline(prompt_embeds=mapping_feature, guidance_scale=guidance_scale,
                                        ).images[0]
            predicted_images_ft.append(cur_predicted_images_ft)

        return pred_out, predicted_images_ft

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        input_images = batch.get('input_images', None)
        if input_images is not None:
            input_images = input_images[0]
        gt_image = batch['output_image'][0]
        input_utterance = batch['source'][0]
        gt_out = batch['target'][0]
        # captions = batch['caption'][0]
        task_name = batch['task_name'][0]

        predicted_images_ft = None
        predicted_images_nl = None
        current_step_prompt = None

        save_dir_cpr = self.output_folder

        if self.encoder_model_config.model_type == 'multimodal_encoder':
            pred_out, predicted_images_ft = self.generate(input_utterance, input_images)

        results = [input_utterance, pred_out, gt_out, predicted_images_ft, predicted_images_nl, gt_image,
                current_step_prompt, task_name]
        return results

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def compute_vae_loss(self, mapping_feature, output_image):
        with torch.no_grad():
            latents = self.vae.encode(output_image).latent_dist.sample()
        mapping_feature = mapping_feature.reshape(-1, 4, 64, 64)
        loss = F.mse_loss(mapping_feature.float(), latents.float(), reduction="mean")
        return loss

    def compute_image_loss(self, mapping_feature, output_image, output_image_feature=None):
        # with torch.no_grad():
        if output_image_feature is not None:
            latents = DiagonalGaussianDistribution(output_image_feature).sample()
        else:
            if len(output_image.shape) == 3:
                output_image = output_image.unsqueeze(0)
            latents = self.vae.encode(output_image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        target = noise
        model_pred = self.unet(noisy_latents, timesteps, mapping_feature).sample
        if self.encoder_model_config.snr_loss:
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                    torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss

    def encode_caption(self, caption, length, inference=False):
        # add_special_tokens = False
        # if len(caption) == 0:
        add_special_tokens = True
        text_inputs = self.sd_tokenizer(
            caption,
            padding="max_length",
            max_length=length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=add_special_tokens
        ).to(self.device2)
        # text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        prompt_embeds = self.sd_text_encoder(**text_inputs)[0]
        return prompt_embeds

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        trainable_param_names = [n for n, p in self.named_parameters() if p.requires_grad]
        # remove untrainable params
        for k in list(checkpoint["state_dict"].keys()):
            if k not in trainable_param_names:
                del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # use pretrained weights for unsaved params
        current_state_dict = self.state_dict()
        state_dict = checkpoint["state_dict"]
        if self.model.using_lora:
            # load lm_head and embed_tokens from pretrained model
            for name in state_dict.keys():
                if "lm_head" in name:
                    for key in current_state_dict.keys():
                        if "lm_head" in key and key != name:
                            current_state_dict[key] = state_dict[name]
                elif "embed_tokens" in name:
                    for key in current_state_dict.keys():
                        if "embed_tokens" in key and key != name:
                            current_state_dict[key] = state_dict[name]
        current_state_dict.update(state_dict)
        checkpoint["state_dict"] = current_state_dict
