# Credits

Every model evaluated here was implemented from its published description, in
this repository, against the shared pipeline interface. No third-party model
code was copied into `qdepipe/`.

Two dependencies are singled out because results are attributed **to** them,
not merely computed with them.

## reservoirpy

The Echo State Network cross-check uses the `reservoirpy` library as an
*independent* implementation of the same method. The point is to show that a
finding is a property of the method rather than of this project's own ESN code:
if both implementations agree, the result is not an artefact of one codebase.

> Trouvain, N., Pedrelli, N., Dinh, T. T., Hinaut, X. (2020).
> *ReservoirPy: an Efficient and User-Friendly Library to Design Echo State
> Networks.* ICANN 2020.
> https://github.com/reservoirpy/reservoirpy — MIT licence.

Used as a pip dependency (`reservoirpy>=0.3.12,<0.4.3`), not vendored.
Call site: `qdepipe/models/reservoirpy_esn.py`.

## Chronos-Bolt

The pretrained zero-shot baseline is the published Chronos-Bolt (small)
checkpoint, evaluated with no training and no tuning on this data. The
checkpoint is downloaded at run time by
`experiments/run_pretrained_zeroshot.py`; no weights are stored here.

> Ansari, A. F. et al. (2024). *Chronos: Learning the Language of Time Series.*
> Transactions on Machine Learning Research.
> Checkpoint: https://huggingface.co/amazon/chronos-bolt-small — Apache 2.0.

## Methods implemented from publications

These were written from the papers, not from the authors' code. Any deviation
is this project's responsibility, and each is scoped as an independent reduced
reimplementation where it is reported.

| Method | Source |
|---|---|
| Next-Generation Reservoir Computing (NG-RC) | Gauthier, Bollt, Griffith, Barbosa (2021), *Next generation reservoir computing*, Nature Communications |
| Echo State Network | Jaeger (2001); practical tuning after Lukoševičius (2012) |
| Extreme Learning Machine | Huang, Zhu, Siew (2006) |
| Recurrence-free quantum reservoir | Ahmed et al. (2024). Some circuit details are left to an appendix figure in the source; the choices made here are disclosed in `experiments/run_rfqrc.py` |
| Dissipative quantum reservoir | Fujii & Nakajima (2017); dissipation-as-a-resource after Sannia et al. (2024) |
| Quantum NG-RC (qNG-RC) | Wang, Sun, Kong, Sun, Zhang (2025), *Physical Review A* 111, 022609. Reimplemented in reduced form; not the authors' full protocol |
| Memory capacity | Jaeger (2001), short-term memory measure |
| Diebold–Mariano test | Diebold & Mariano (1995), with the Harvey, Leybourne & Newbold (1997) small-sample correction |
| Model Confidence Set | Hansen, Lunde & Nason (2011) |
| Holm–Bonferroni correction | Holm (1979) |

## Scientific Python stack

numpy, pandas, scipy, scikit-learn, matplotlib, statsmodels, PyTorch, XGBoost,
LightGBM, pyarrow, DuckDB, pandera, pydantic, pytest. Exact versions in
`requirements.lock`; the toolchain recorded alongside the results is in
`versions.txt`.

## Benchmark systems

All four are generated here by seeded integrators in `qdepipe/data/`. No
downloaded datasets.

| System | Source |
|---|---|
| Hénon map | Hénon (1976) |
| Lorenz-63 | Lorenz (1963) |
| Mackey–Glass (MG-17) | Mackey & Glass (1977) |
| Lorenz-96 | Lorenz (1996) |
