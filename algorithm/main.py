from bootstrap import run_bootstrap_experiments
from co_teaching import run_coteaching_experiments
from forward import run_forward_experiments
from create_charts import create_charts

RUN_EXPERIMENTS = True
NUM_RUNS = 3
OUTPUT_PATH = "results/"

CREATE_CHARTS = False


if __name__ == "__main__":

    if RUN_EXPERIMENTS:
        # Run Bootstrap Experiments
        run_bootstrap_experiments(num_runs=NUM_RUNS, output_path=OUTPUT_PATH)

        # Run Co-Teaching Experiments
        run_coteaching_experiments(num_runs=NUM_RUNS, output_path=OUTPUT_PATH)

        # Run Forward Experiments
        run_forward_experiments(num_runs=NUM_RUNS, output_path=OUTPUT_PATH)

    if CREATE_CHARTS:
        create_charts()