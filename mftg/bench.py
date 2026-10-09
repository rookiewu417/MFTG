# Synthetic step-time benchmark of the MFTG model (same shapes as training: 2500 stocks x 10 days x 48 bars, lookback 60).
import sys, time, torch
sys.path.insert(0, '/content/code')
from train_mftg import MFTG, corr_loss
dev = torch.device('cuda')
for cin, name in [(5, 'intra'), (8, 'intra+flow')]:
    m = MFTG(32, 64, 2, True, cin=cin).to(dev); opt = torch.optim.AdamW(m.parameters(), 5e-4)
    d = torch.randn(2500, 60, 10, device=dev) * 0.02; it = torch.randn(2500, 10, 48, cin, device=dev) * 0.02; y = torch.randn(2500, device=dev)
    for i in range(25):
        if i == 5: torch.cuda.synchronize(); t = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16): p = m(d, it)
        loss = corr_loss(p.float(), y, torch.ones_like(y)); opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); print(name, torch.cuda.get_device_name(), f'{(time.time() - t) / 20 * 1000:.0f} ms/step', flush=True)
