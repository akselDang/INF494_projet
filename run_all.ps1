# All MNIST experiments, 3 seeds each, then the summary table and the figures (Windows PowerShell).
# Stops at the first failing command.
function Run { python @args; if ($LASTEXITCODE -ne 0) { throw "failed: python $args" } }

foreach ($seed in 0, 1, 2) {
    Run -m src.train --dataset mnist --model gcn --features 3 --seed $seed
    Run -m src.train --dataset mnist --model gcn  --seed $seed
    Run -m src.train --dataset mnist --model sage --seed $seed
    Run -m src.train --dataset mnist --model sage --raw-features --seed $seed
    Run -m src.train --dataset mnist --model lenet --seed $seed
    Run -m src.train --dataset mnist --model cnn  --seed $seed
}
Run -m src.summarize
Run report/make_numbers.py
Run -m src.visualize --dataset mnist --models cnn lenet sage gcn --seed 0
Run -m src.demo_pipeline --stats
