# CNF-DampSine environment and reference audit

Official CNF source checked at commit `9af9a830dd90f0679f30532c0c4edd53c36eca12` (GitHub `zfc946/CNF`). The official repository is a reference, not imported into this project yet.

| Item | Official source | Project decision |
| --- | --- | --- |
| Gaussian | `network.py` factory key `kernel`; `nbf.KernelBasis(sample_pt, encoder, num_basis, dim, kernel_type="Gaussian", init_sigma, per_basis_sigma)` | MVP will fix sigma at 0.05; official `KernelBasis` always declares it as a trainable parameter, so the adapter must explicitly freeze it. |
| Positional encoder | `network.py` uses `Encoder(PositionalEncoder(dim[0], n_freqs=3))` | Preserve this structure for CNF-Gaussian. |
| CNF-SIREN | `network.py` factory key `hypernet_siren`, `Siren(..., hidden_features=21, hidden_layers=1, outermost_linear=True)` wrapped by `nbf.HypernetBasis` | Preserve the basis-ID-conditioned hypernetwork route. |
| Alternate SIREN | `sepbasis_siren` creates one `Siren` per constraint and wraps `nbf.SeparateBasis` | Keep as a separate ablation. |
| Hard solve | `nbf.py` uses `torch.linalg.solve` | Keep direct square solve; rectangular extensions have separate semantics. |

The current remote host reports Python 3.10.12, PyTorch 2.5.1+cu124, and pytest 9.2.0. The plan's reference stack is Python 3.9, PyTorch 1.13.0, PyTorch Lightning 1.8.1, and SciPy 1.9.3. Initial core tests run under the available stack; reference-stack compatibility has not been verified. The remote host cannot reach GitHub through its configured proxy, so the official source was inspected locally.
