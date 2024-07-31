from .METEOR import calculate_meteor
from .BLEU import calculate_bleu
from .CIDEr import calculate_cider
from .ImageSequence import calculate_IS_and_FID
from .SSIM import calculate_ssim
from .PSNR import calculate_psnr
from .gpt4o_eval import gpt4o_eval_task4, gpt4o_eval_task1, gpt4o_eval_task2
from .LPIPS import calculate_lpips

split_token = "<IMAGE>"

