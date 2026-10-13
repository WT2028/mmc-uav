"""Calculate design criteria, replay measured multibody tests, or sample the coupling map."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .mmc_analysis import analytical, mechanism, coupling, replay

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--multibody', action='store_true', help='Replay the thirteen measured 0.35 s tests')
    parser.add_argument('--map', action='store_true', help='Calculate the specified S/T map grid')
    parser.add_argument('--grid', type=int, default=21)
    parser.add_argument('--output', type=Path, default=ROOT/'results/mechanisms')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = {'mechanism':mechanism.self_test(), 'replay':replay.self_test()}
    coupling.self_test()
    c90 = analytical.t90_dimensionless(1.2)
    _, cv = analytical.peak_velocity_dimensionless(1.2)
    criteria = {'zeta':1.2,'C90':c90,'Cv':cv,'t90_20_rad_s':c90/20.,
                'ideal_50mm_bandwidth_interval_rad_s':[c90/.35,.25/(cv*.05)],
                'cog_10mm_min_mu_working':.01/(2*.135),
                'cog_10mm_min_mu_mechanical':.01/(2*.15),
                'attitude_task':mechanism.attitude_task(), 'checks':checks}
    (args.output/'design_criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
    trajectories = []
    for wn in [10.,20.,30.]:
        for row in mechanism.constrained_step(wn,.05,.05,free_body=True):
            trajectories.append({'wn_rad_s':wn,**row})
    with (args.output/'step_responses.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(trajectories[0]))
        writer.writeheader();writer.writerows(trajectories)
    if args.multibody:
        cases = json.loads((ROOT/'data/multibody/index.json').read_text())
        rows = []
        for case in cases:
            history = replay.PhysicalHistory(ROOT/'data/multibody'/case['case'])
            run = replay.replay(history,aerodynamic='nominal')
            summary = replay.summarize(history,run,.001,'nominal')
            rows.append({'case':case['case'],'MF_plant':summary['MF_plant'],
                         'passed':summary['MF_plant_budget_pass'],
                         'B':summary['B'],'omissions':summary['omissions_vs_MF']})
            print(json.dumps({'case':case['case'],'passed':summary['MF_plant_budget_pass']}),flush=True)
        result = {'cases':rows,'all_passed':all(r['passed'] for r in rows),
                  'max_position_error_mm':1000*max(r['MF_plant']['Ep_m'] for r in rows),
                  'max_attitude_error_deg':max(r['MF_plant']['Ea_deg'] for r in rows)}
        (args.output/'multibody.json').write_text(json.dumps(result,indent=2)+'\n')
        if not result['all_passed']:
            raise SystemExit('A multibody error budget was exceeded')
    if args.map:
        if args.grid < 2:
            parser.error('--grid must be at least 2')
        rows=[]
        for s in np.linspace(.001,.135,args.grid):
            for duration in np.geomspace(.03,.6,args.grid):
                case=coupling.family_case(float(s),float(duration))
                rows.append(coupling.summarize(case,coupling.simulate(case),'cad',.001))
        with (args.output/'coupling_map.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(args.output)


if __name__ == '__main__':
    main()
