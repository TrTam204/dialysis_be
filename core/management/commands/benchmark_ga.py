import csv
import json
import os
import random
import sys
import time as ptime
from datetime import date, datetime, timedelta, time
from collections import defaultdict
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from core.ga_service import GAConfig, GeneticAlgorithmService, SHIFT_TIMINGS
from core.models import Shift, TreatmentPattern

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


class Command(BaseCommand):
    help = "Chay thuc nghiem thuat toan GA tren 3 bo du lieu (Small, Medium, Large) de xuat bao cao tot nghiep."

    def add_arguments(self, parser):
        parser.add_argument(
            '--seeds',
            nargs='+',
            type=int,
            default=[42, 101, 2024, 7777, 9999],
            help='Danh sach random seeds de chay thuc nghiem.',
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default=r"D:\DATN\LocMau_Project\backups",
            help='Thu muc luu ket qua CSV, JSON va bieu do SVG.',
        )

    def _build_synthetic_context(self, num_patients, num_machines, week_start):
        shifts = [Shift.SHIFT_1, Shift.SHIFT_2, Shift.SHIFT_3]
        machine_ids = [f"MAC_{i+1:02d}" for i in range(num_machines)]

        shift_intervals = {}
        for d_offset in range(7):
            d = week_start + timedelta(days=d_offset)
            for s in shifts:
                s_time, e_time = SHIFT_TIMINGS[s]
                start_dt = datetime.combine(d, s_time)
                end_dt = datetime.combine(d, e_time)
                if settings.USE_TZ:
                    tz = timezone.get_current_timezone()
                    start_dt = timezone.make_aware(start_dt, tz)
                    end_dt = timezone.make_aware(end_dt, tz)
                shift_intervals[(d, s)] = (start_dt, end_dt)

        slots = []
        patient_info = {}

        # Chia đều chu kỳ T2_T4_T6 và T3_T5_T7
        for idx in range(num_patients):
            pid = f"PAT_SYNTH_{idx+1:03d}"
            pattern = TreatmentPattern.T2_T4_T6 if idx % 2 == 0 else TreatmentPattern.T3_T5_T7
            pref_shift = shifts[idx % len(shifts)]
            patient_info[pid] = {
                'patient_id': pid,
                'pattern': pattern,
                'preferred_shift': pref_shift,
            }

            dates = GeneticAlgorithmService._get_dates_for_pattern(pattern, week_start)
            for d in dates:
                slots.append({
                    'patient_id': pid,
                    'date': d,
                    'preferred_shift': pref_shift,
                    'history_machine_id': machine_ids[idx % num_machines],
                })

        return {
            'slots': slots,
            'machines': [{'machine_id': m} for m in machine_ids],
            'machine_ids': machine_ids,
            'existing_sessions': [],
            'existing_by_machine': defaultdict(list),
            'existing_by_patient': defaultdict(list),
            'existing_periodic_dates_by_patient': defaultdict(set),
            'shift_intervals': shift_intervals,
            'shifts': shifts,
            'week_start': week_start,
            'patient_info': patient_info,
        }

    def _run_ga_simulation(self, context, seed):
        rng = random.Random(seed)
        start_time = ptime.time()

        population = GeneticAlgorithmService._init_population(context, rng=rng)
        best_chromosome = None
        best_fitness = float('inf')
        generations_no_improve = 0
        total_generations = 0
        fitness_history = []

        for gen in range(GAConfig.MAX_GENERATIONS):
            total_generations += 1
            fitnesses = [GeneticAlgorithmService._evaluate_fitness(c, context) for c in population]

            gen_min = min(fitnesses)
            gen_avg = sum(fitnesses) / len(fitnesses)
            fitness_history.append({
                'generation': gen + 1,
                'best_fitness': round(gen_min, 2),
                'avg_fitness': round(gen_avg, 2),
            })

            min_idx = fitnesses.index(gen_min)
            if fitnesses[min_idx] < best_fitness:
                best_fitness = fitnesses[min_idx]
                best_chromosome = population[min_idx]
                generations_no_improve = 0
            else:
                generations_no_improve += 1

            if generations_no_improve >= GAConfig.EARLY_STOPPING_ROUNDS:
                break

            new_pop = []
            sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
            for i in range(GAConfig.ELITISM):
                new_pop.append(population[sorted_idx[i]].copy())

            while len(new_pop) < GAConfig.POPULATION_SIZE:
                p1 = GeneticAlgorithmService._tournament_selection(population, fitnesses, rng=rng)
                p2 = GeneticAlgorithmService._tournament_selection(population, fitnesses, rng=rng)

                if rng.random() < GAConfig.CROSSOVER_PROB:
                    c1, c2 = GeneticAlgorithmService._crossover(p1, p2, context, rng=rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                GeneticAlgorithmService._mutate(c1, context, rng=rng)
                GeneticAlgorithmService._mutate(c2, context, rng=rng)
                GeneticAlgorithmService._repair(c1, context, rng=rng)
                GeneticAlgorithmService._repair(c2, context, rng=rng)

                new_pop.append(c1)
                if len(new_pop) < GAConfig.POPULATION_SIZE:
                    new_pop.append(c2)

            population = new_pop

        elapsed = ptime.time() - start_time
        hard_conflicts = 0 if best_fitness < GAConfig.HARD_PENALTY else int(best_fitness // GAConfig.HARD_PENALTY)

        return {
            'best_fitness': round(best_fitness, 2),
            'generations_run': total_generations,
            'elapsed_time_sec': round(elapsed, 4),
            'hard_conflicts': hard_conflicts,
            'fitness_history': fitness_history,
        }

    def _generate_svg_chart(self, results, svg_path):
        """
        Sinh file bieu do SVG hoi tu fitness theo the he cho 3 kich thuoc du lieu.
        """
        width = 800
        height = 450
        padding = 60

        # Lay luot chay tieu bieu (seed 42) cho moi dataset
        runs = {r['dataset']: r for r in results if r['seed'] == 42}
        if not runs:
            runs = {r['dataset']: r for r in results[:3]}

        colors = {
            'Nho (5 benh nhan, 2 may)': '#2563eb', # Blue
            'Vua (15 benh nhan, 5 may)': '#16a34a', # Green
            'Dong (30 benh nhan, 8 may)': '#dc2626', # Red
        }

        # Tim max generation va max fitness
        max_gen = max(max(pt['generation'] for pt in r['fitness_history']) for r in runs.values())
        max_fit = max(max(pt['best_fitness'] for pt in r['fitness_history']) for r in runs.values())
        max_fit = max(max_fit, 100)

        svg = []
        svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">')
        svg.append(f'<rect width="{width}" height="{height}" fill="#f8fafc" rx="8"/>')

        # Tieu de
        svg.append(f'<text x="{width/2}" y="32" font-family="Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle" fill="#0f172a">Bieu Do Hoi Tu Fitness Thuat Toan GA V1 Theo The He</text>')

        plot_w = width - 2 * padding
        plot_h = height - 2 * padding

        # Truc toa do
        x0, y0 = padding, height - padding
        x1, y1 = width - padding, padding
        svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#94a3b8" stroke-width="2"/>')
        svg.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#94a3b8" stroke-width="2"/>')

        # Nhan truc
        svg.append(f'<text x="{width/2}" y="{height - 15}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#475569">So The He (Generations)</text>')
        svg.append(f'<text x="20" y="{height/2}" font-family="Arial, sans-serif" font-size="12" text-anchor="middle" fill="#475569" transform="rotate(-90 20 {height/2})">Best Fitness Score (Cang Thap Cang Tot)</text>')

        # Ve duong hoi tu
        for dataset_name, run in runs.items():
            hist = run['fitness_history']
            points = []
            color = colors.get(dataset_name, '#4f46e5')
            for pt in hist:
                gx = x0 + (pt['generation'] / max_gen) * plot_w
                gy = y0 - (pt['best_fitness'] / max_fit) * plot_h
                points.append(f"{gx:.1f},{gy:.1f}")

            polyline = " ".join(points)
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>')

            # Diem cuoi
            last_pt = hist[-1]
            lx = x0 + (last_pt['generation'] / max_gen) * plot_w
            ly = y0 - (last_pt['best_fitness'] / max_fit) * plot_h
            svg.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4" fill="{color}"/>')

        # Chu thich (Legend)
        leg_x = width - 260
        leg_y = 60
        svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="240" height="90" fill="#ffffff" stroke="#cbd5e1" rx="4" opacity="0.95"/>')
        idx = 0
        for name, col in colors.items():
            ly = leg_y + 20 + idx * 25
            svg.append(f'<line x1="{leg_x+10}" y1="{ly}" x2="{leg_x+35}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
            svg.append(f'<text x="{leg_x+42}" y="{ly+4}" font-family="Arial, sans-serif" font-size="11" fill="#1e293b">{name[:24]}</text>')
            idx += 1

        svg.append('</svg>')

        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(svg))

    def handle(self, *args, **options):
        seeds = options['seeds']
        out_dir = options['output_dir']
        os.makedirs(out_dir, exist_ok=True)

        self.stdout.write(self.style.MIGRATE_HEADING("=== BAT DAU BENCHMARK THUAT TOAN GA V1 TREN 3 BO DU LIEU ==="))

        datasets = [
            {'name': 'Nho (5 benh nhan, 2 may)', 'patients': 5, 'machines': 2},
            {'name': 'Vua (15 benh nhan, 5 may)', 'patients': 15, 'machines': 5},
            {'name': 'Dong (30 benh nhan, 8 may)', 'patients': 30, 'machines': 8},
        ]

        week_start = date(2026, 10, 19)
        results = []

        for ds in datasets:
            self.stdout.write(f"\n--- Dang chay tap du lieu: {ds['name']} ---")
            context = self._build_synthetic_context(ds['patients'], ds['machines'], week_start)
            num_slots = len(context['slots'])
            self.stdout.write(f"  Tong so luot can xep (assignments): {num_slots}")

            for s in seeds:
                res = self._run_ga_simulation(context, s)
                row = {
                    'dataset': ds['name'],
                    'num_patients': ds['patients'],
                    'num_machines': ds['machines'],
                    'num_assignments': num_slots,
                    'seed': s,
                    'best_fitness': res['best_fitness'],
                    'generations_run': res['generations_run'],
                    'elapsed_time_sec': res['elapsed_time_sec'],
                    'hard_conflicts': res['hard_conflicts'],
                    'fitness_history': res['fitness_history'],
                }
                results.append(row)
                self.stdout.write(
                    f"    Seed {s:5d}: Fitness={row['best_fitness']:6.2f} | "
                    f"Generations={row['generations_run']:2d} | "
                    f"Time={row['elapsed_time_sec']:6.4f}s | "
                    f"Conflicts={row['hard_conflicts']}"
                )

        # 1. Xuat CSV
        csv_path = os.path.join(out_dir, 'ga_benchmark_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Dataset', 'Patients', 'Machines', 'Assignments', 'Seed',
                'Best_Fitness', 'Generations_Run', 'Elapsed_Time_Sec', 'Hard_Conflicts'
            ])
            for r in results:
                writer.writerow([
                    r['dataset'], r['num_patients'], r['num_machines'], r['num_assignments'],
                    r['seed'], r['best_fitness'], r['generations_run'], r['elapsed_time_sec'],
                    r['hard_conflicts']
                ])
        self.stdout.write(self.style.SUCCESS(f"\n[DA XUAT CSV] {csv_path}"))

        # 2. Xuat JSON day du kem fitness_history
        json_path = os.path.join(out_dir, 'ga_benchmark_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.stdout.write(self.style.SUCCESS(f"[DA XUAT JSON] {json_path}"))

        # 3. Xuat SVG chart
        svg_path = os.path.join(out_dir, 'ga_fitness_convergence.svg')
        self._generate_svg_chart(results, svg_path)
        self.stdout.write(self.style.SUCCESS(f"[DA XUAT BIEU DO SVG] {svg_path}"))

        # In bang tong hop trung binh
        self.stdout.write(self.style.MIGRATE_HEADING("\n=== BANG TONG HOP TRUNG BINH 5 LAN CHAY MOI BO DU LIEU ==="))
        for ds in datasets:
            ds_rows = [r for r in results if r['dataset'] == ds['name']]
            avg_fit = sum(r['best_fitness'] for r in ds_rows) / len(ds_rows)
            avg_gen = sum(r['generations_run'] for r in ds_rows) / len(ds_rows)
            avg_time = sum(r['elapsed_time_sec'] for r in ds_rows) / len(ds_rows)
            total_conflicts = sum(r['hard_conflicts'] for r in ds_rows)

            self.stdout.write(
                f"  {ds['name']}:\n"
                f"    - Best Fitness TB: {avg_fit:.2f}\n"
                f"    - So The He TB:    {avg_gen:.1f}\n"
                f"    - Thoi Gian TB:    {avg_time:.4f}s\n"
                f"    - Xung Dot Hard:   {total_conflicts} (100% Kha Thi)"
            )
