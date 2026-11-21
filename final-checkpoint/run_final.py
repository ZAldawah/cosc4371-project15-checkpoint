"""Reproduce all final-checkpoint experiments and generated deliverables."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=HERE/'data')
    parser.add_argument('--reuse-distilbert',action='store_true',help='Use locally saved fine-tuned weights instead of training again')
    args=parser.parse_args()
    data=['--data-dir',str(args.data_dir.resolve())]
    commands=[['verify_data.py',*data],['run_classical.py',*data],
              ['run_transformers.py',*data]+(['--reuse-distilbert'] if args.reuse_distilbert else []),
              ['run_zeroshot.py',*data],['assemble_results.py'],['build_final_report.py'],['verify_final.py']]
    for command in commands:
        print('Running: '+' '.join(command),flush=True)
        subprocess.run([sys.executable,str(HERE/command[0]),*command[1:]],cwd=HERE,check=True)

if __name__=='__main__':
    main()
