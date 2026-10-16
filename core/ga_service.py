import math
import random
import statistics
import time as ptime
from datetime import timedelta, time, datetime
from collections import defaultdict
from django.conf import settings
from django.db.models import Q
from django.utils import timezone
from core.models import Patient, DialysisMachine, DialysisSession, SchedulePlan, ScheduleAssignment, Shift
from django.db import transaction
import logging

logger = logging.getLogger(__name__)

class GAConfig:
    POPULATION_SIZE = 50
    MAX_GENERATIONS = 100
    TOURNAMENT_SIZE = 3
    ELITISM = 2
    CROSSOVER_PROB = 0.8
    MUTATION_PROB = 0.1
    EARLY_STOPPING_ROUNDS = 20
    
    HARD_PENALTY = 10000
    SOFT_PREFERRED_SHIFT = 100
    SOFT_STABILITY = 20
    SOFT_LOAD_STD_MULTIPLIER = 10

SHIFT_TIMINGS = {
    Shift.SHIFT_1: (time(7, 0), time(11, 0)),
    Shift.SHIFT_2: (time(12, 0), time(16, 0)),
    Shift.SHIFT_3: (time(17, 0), time(21, 0)),
}

class GeneticAlgorithmService:
    @classmethod
    def generate_plan(cls, department_id, week_start_date, user, seed=None):
        start_time = ptime.time()
        rng = random.Random(seed) if seed is not None else random.Random()
        
        # 1. Prepare Context (scoped to department_id)
        context = cls._build_context(department_id, week_start_date)
        if not context['slots']:
            return None, "Không tìm thấy bệnh nhân hợp lệ cần phân ca trong khoa."
        if not context['machines']:
            return None, "Không tìm thấy máy lọc máu nào đang hoạt động."

        # 2. Init Population
        population = cls._init_population(context, rng=rng)
        
        best_chromosome = None
        best_fitness = float('inf')
        generations_no_improve = 0
        total_generations = 0
        fitness_history = []
        
        # 3. Evolution Loop
        for gen in range(GAConfig.MAX_GENERATIONS):
            total_generations += 1
            # Evaluate Fitness
            fitnesses = [cls._evaluate_fitness(c, context) for c in population]
            
            gen_min = min(fitnesses)
            gen_avg = sum(fitnesses) / len(fitnesses)
            fitness_history.append({
                'generation': gen + 1,
                'best_fitness': round(gen_min, 2),
                'avg_fitness': round(gen_avg, 2),
            })

            # Find Best
            min_fit_idx = fitnesses.index(gen_min)
            if fitnesses[min_fit_idx] < best_fitness:
                best_fitness = fitnesses[min_fit_idx]
                best_chromosome = population[min_fit_idx]
                generations_no_improve = 0
            else:
                generations_no_improve += 1
                
            if generations_no_improve >= GAConfig.EARLY_STOPPING_ROUNDS:
                break
                
            # Selection
            new_population = []
            
            # Elitism
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
            for i in range(GAConfig.ELITISM):
                new_population.append(population[sorted_indices[i]].copy())
                
            # Fill the rest
            while len(new_population) < GAConfig.POPULATION_SIZE:
                p1 = cls._tournament_selection(population, fitnesses, rng=rng)
                p2 = cls._tournament_selection(population, fitnesses, rng=rng)
                
                # Crossover
                if rng.random() < GAConfig.CROSSOVER_PROB:
                    c1, c2 = cls._crossover(p1, p2, context, rng=rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                cls._mutate(c1, context, rng=rng)
                cls._mutate(c2, context, rng=rng)
                
                # Validate & Repair
                cls._repair(c1, context, rng=rng)
                cls._repair(c2, context, rng=rng)
                
                new_population.append(c1)
                if len(new_population) < GAConfig.POPULATION_SIZE:
                    new_population.append(c2)
                    
            population = new_population

        elapsed = ptime.time() - start_time
        
        # 4. Check for Hard Violations
        if best_fitness >= GAConfig.HARD_PENALTY:
            return None, "Không thể tạo lịch hợp lệ với các máy và ràng buộc hiện tại."

        # 5. Save to DB with metadata
        plan = cls._save_chromosome(
            best_chromosome, best_fitness, context, department_id, week_start_date,
            user, total_generations, elapsed, seed=seed, fitness_history=fitness_history
        )
        return plan, None

    @classmethod
    def _get_shift_interval(cls, date_val, shift_val):
        start_time, end_time = SHIFT_TIMINGS[shift_val]
        start_dt = datetime.combine(date_val, start_time)
        end_dt = datetime.combine(date_val, end_time)
        if settings.USE_TZ:
            tz = timezone.get_current_timezone()
            if timezone.is_naive(start_dt):
                start_dt = timezone.make_aware(start_dt, tz)
            if timezone.is_naive(end_dt):
                end_dt = timezone.make_aware(end_dt, tz)
        return start_dt, end_dt

    @classmethod
    def _build_context(cls, department_id, week_start):
        week_end = week_start + timedelta(days=6)
        
        tz = timezone.get_current_timezone() if settings.USE_TZ else None
        start_bound = datetime.combine(week_start, time.min)
        end_bound = datetime.combine(week_end, time.max)
        if settings.USE_TZ:
            start_bound = timezone.make_aware(start_bound, tz)
            end_bound = timezone.make_aware(end_bound, tz)

        # 1. GA chỉ lấy Patient có department_id trùng với khoa được lập lịch
        patients = list(Patient.objects.filter(
            department_id=department_id,
            status__in=[Patient.Status.ACTIVE, Patient.Status.IN_TREATMENT, Patient.Status.STABLE],
            treatment_pattern__isnull=False
        ))
        
        # 2. Chỉ sử dụng máy AVAILABLE thuộc đúng khoa
        machines = list(DialysisMachine.objects.filter(
            department_id=department_id,
            status=DialysisMachine.Status.AVAILABLE
        ))
        
        machine_ids = [m.machine_id for m in machines]
        
        shifts = [Shift.SHIFT_1, Shift.SHIFT_2, Shift.SHIFT_3]
        shift_intervals = {}
        for d_offset in range(7):
            d = week_start + timedelta(days=d_offset)
            for s in shifts:
                shift_intervals[(d, s)] = cls._get_shift_interval(d, s)

        existing_sessions = list(DialysisSession.objects.filter(
            machine__department_id=department_id,
            scheduled_start__lt=end_bound,
            scheduled_end__gt=start_bound,
            status__in=[DialysisSession.Status.SCHEDULED, DialysisSession.Status.IN_PROGRESS, DialysisSession.Status.COMPLETED]
        ))

        existing_by_machine = defaultdict(list)
        for sess in existing_sessions:
            s_start = sess.scheduled_start
            s_end = sess.scheduled_end
            if settings.USE_TZ:
                if timezone.is_naive(s_start):
                    s_start = timezone.make_aware(s_start, tz)
                if timezone.is_naive(s_end):
                    s_end = timezone.make_aware(s_end, tz)
            existing_by_machine[sess.machine_id].append({
                'session_id': sess.session_id,
                'start': s_start,
                'end': s_end,
            })

        candidate_patient_ids = [p.patient_id for p in patients]
        existing_patient_sessions = list(DialysisSession.objects.filter(
            patient_id__in=candidate_patient_ids,
            scheduled_start__lt=end_bound,
            scheduled_end__gt=start_bound,
        ).exclude(status=DialysisSession.Status.CANCELLED))

        existing_by_patient = defaultdict(list)
        existing_periodic_dates_by_patient = defaultdict(set)

        for sess in existing_patient_sessions:
            s_start = sess.scheduled_start
            s_end = sess.scheduled_end
            if settings.USE_TZ:
                if timezone.is_naive(s_start):
                    s_start = timezone.make_aware(s_start, tz)
                if timezone.is_naive(s_end):
                    s_end = timezone.make_aware(s_end, tz)

            sess_local_date = timezone.localtime(s_start, tz).date() if settings.USE_TZ else s_start.date()
            existing_by_patient[sess.patient_id].append({
                'session_id': sess.session_id,
                'start': s_start,
                'end': s_end,
                'date': sess_local_date,
            })
            if sess.schedule_assignment_id is not None:
                existing_periodic_dates_by_patient[sess.patient_id].add(sess_local_date)
        
        # historical machine for stability
        patient_history = {}
        for p in patients:
            last_sess = DialysisSession.objects.filter(
                patient=p,
                scheduled_start__lt=start_bound
            ).order_by('-scheduled_start').first()
            if last_sess:
                patient_history[p.patient_id] = last_sess.machine_id

        slots = []
        patient_info = {}
        
        for p in patients:
            patient_info[p.patient_id] = p
            dates = cls._get_dates_for_pattern(p.treatment_pattern, week_start)
            for d in dates:
                slots.append({
                    'patient_id': p.patient_id,
                    'date': d,
                    'preferred_shift': p.preferred_shift,
                    'history_machine_id': patient_history.get(p.patient_id)
                })
                
        return {
            'slots': slots,
            'machines': machines,
            'machine_ids': machine_ids,
            'existing_sessions': existing_sessions,
            'existing_by_machine': existing_by_machine,
            'existing_by_patient': existing_by_patient,
            'existing_periodic_dates_by_patient': existing_periodic_dates_by_patient,
            'shift_intervals': shift_intervals,
            'shifts': shifts,
            'week_start': week_start,
            'patient_info': patient_info
        }

    @classmethod
    def _get_dates_for_pattern(cls, pattern, week_start):
        # T2_T4_T6: Monday=0, Wednesday=2, Friday=4
        if pattern == 'T2_T4_T6':
            return [week_start + timedelta(days=0), week_start + timedelta(days=2), week_start + timedelta(days=4)]
        elif pattern == 'T3_T5_T7':
            return [week_start + timedelta(days=1), week_start + timedelta(days=3), week_start + timedelta(days=5)]
        return []

    @classmethod
    def _init_population(cls, context, rng=None):
        _rng = rng if rng is not None else random
        population = []
        for _ in range(GAConfig.POPULATION_SIZE):
            chromosome = []
            
            # Group slots by date to avoid machine duplication heuristically
            slots_by_date = defaultdict(list)
            for slot in context['slots']:
                slots_by_date[slot['date']].append(slot)
                
            for date_val, daily_slots in slots_by_date.items():
                daily_slots_copy = list(daily_slots)
                _rng.shuffle(daily_slots_copy)
                
                # Keep track of used (shift, machine) on this date
                used = set()
                
                for slot in daily_slots_copy:
                    shift = slot['preferred_shift'] if slot['preferred_shift'] and _rng.random() > 0.3 else _rng.choice(context['shifts'])
                    
                    available_machines = [m for m in context['machine_ids'] if (shift, m) not in used]
                    
                    if not available_machines:
                        # Fallback: pick any shift that has available machines
                        for s in context['shifts']:
                            available_machines = [m for m in context['machine_ids'] if (s, m) not in used]
                            if available_machines:
                                shift = s
                                break
                    
                    if available_machines:
                        machine_id = _rng.choice(available_machines)
                        used.add((shift, machine_id))
                    else:
                        # Force assign (will be penalized later)
                        machine_id = _rng.choice(context['machine_ids'])
                        
                    chromosome.append({
                        'patient_id': slot['patient_id'],
                        'date': date_val,
                        'shift': shift,
                        'machine_id': machine_id
                    })
            
            population.append(chromosome)
        return population

    @classmethod
    def _evaluate_fitness(cls, chromosome, context):
        penalty = 0
        used_slots = set()
        machine_load = defaultdict(int)
        patient_daily_assignments = defaultdict(int)
        
        for gene in chromosome:
            # 1. Hard: duplicate machine/shift/date
            key = (gene['date'], gene['shift'], gene['machine_id'])
            if key in used_slots:
                penalty += GAConfig.HARD_PENALTY
            used_slots.add(key)
            
            # 2. Hard: tối đa 1 phiên/bệnh nhân/ngày trong lịch GA
            patient_day_key = (gene['patient_id'], gene['date'])
            patient_daily_assignments[patient_day_key] += 1
            if patient_daily_assignments[patient_day_key] > 1:
                penalty += GAConfig.HARD_PENALTY

            gene_start, gene_end = context['shift_intervals'][(gene['date'], gene['shift'])]

            # 3. Hard: overlap with existing session on machine
            machine_sessions = context.get('existing_by_machine', {}).get(gene['machine_id'])
            if machine_sessions:
                for sess in machine_sessions:
                    if max(gene_start, sess['start']) < min(gene_end, sess['end']):
                        penalty += GAConfig.HARD_PENALTY
                        break

            # 4. Hard: overlap with existing session of the same patient
            patient_sessions = context.get('existing_by_patient', {}).get(gene['patient_id'])
            if patient_sessions:
                for sess in patient_sessions:
                    if max(gene_start, sess['start']) < min(gene_end, sess['end']):
                        penalty += GAConfig.HARD_PENALTY
                        break

            # 5. Hard: xung đột nếu bệnh nhân đã có phiên định kỳ trong ngày (Asia/Ho_Chi_Minh)
            periodic_dates = context.get('existing_periodic_dates_by_patient', {}).get(gene['patient_id'], set())
            if gene['date'] in periodic_dates:
                penalty += GAConfig.HARD_PENALTY
                        
            # Soft constraints
            # Find original slot context
            slot_ctx = next(s for s in context['slots'] if s['patient_id'] == gene['patient_id'] and s['date'] == gene['date'])
            
            # Preferred shift
            if slot_ctx['preferred_shift'] and gene['shift'] != slot_ctx['preferred_shift']:
                penalty += GAConfig.SOFT_PREFERRED_SHIFT
                
            # Stability
            if slot_ctx['history_machine_id'] and gene['machine_id'] != slot_ctx['history_machine_id']:
                penalty += GAConfig.SOFT_STABILITY
                
            machine_load[gene['machine_id']] += 1
            
        # Load balancing
        if len(machine_load) > 0:
            loads = list(machine_load.values())
            # For machines not used, load is 0
            for m in context['machine_ids']:
                if m not in machine_load:
                    loads.append(0)
            if len(loads) > 1:
                std_dev = statistics.stdev(loads)
                penalty += std_dev * GAConfig.SOFT_LOAD_STD_MULTIPLIER
                
        return penalty

    @classmethod
    def _tournament_selection(cls, population, fitnesses, rng=None):
        _rng = rng if rng is not None else random
        best = None
        best_fit = float('inf')
        for _ in range(GAConfig.TOURNAMENT_SIZE):
            idx = _rng.randint(0, len(population) - 1)
            if fitnesses[idx] < best_fit:
                best_fit = fitnesses[idx]
                best = population[idx]
        return best

    @classmethod
    def _crossover(cls, p1, p2, context, rng=None):
        _rng = rng if rng is not None else random
        # Day-based crossover
        c1, c2 = [], []
        # Group by date
        p1_by_date = defaultdict(list)
        p2_by_date = defaultdict(list)
        for g in p1: p1_by_date[g['date']].append(g)
        for g in p2: p2_by_date[g['date']].append(g)
        
        dates = list(p1_by_date.keys())
        for d in dates:
            if _rng.random() < 0.5:
                c1.extend([dict(g) for g in p1_by_date[d]])
                c2.extend([dict(g) for g in p2_by_date[d]])
            else:
                c1.extend([dict(g) for g in p2_by_date[d]])
                c2.extend([dict(g) for g in p1_by_date[d]])
                
        return c1, c2

    @classmethod
    def _mutate(cls, chromosome, context, rng=None):
        _rng = rng if rng is not None else random
        if _rng.random() > GAConfig.MUTATION_PROB:
            return
            
        # Swap mutation within same date
        dates = list(set(g['date'] for g in chromosome))
        target_date = _rng.choice(dates)
        genes_in_date = [g for g in chromosome if g['date'] == target_date]
        if len(genes_in_date) >= 2:
            g1, g2 = _rng.sample(genes_in_date, 2)
            # Swap (shift, machine_id)
            temp_s, temp_m = g1['shift'], g1['machine_id']
            g1['shift'], g1['machine_id'] = g2['shift'], g2['machine_id']
            g2['shift'], g2['machine_id'] = temp_s, temp_m

    @classmethod
    def _repair(cls, chromosome, context, rng=None):
        _rng = rng if rng is not None else random
        # Check and fix hard constraints within chromosome (duplicate machine per shift)
        by_date = defaultdict(list)
        for g in chromosome:
            by_date[g['date']].append(g)
            
        for date_val, genes in by_date.items():
            used = set()
            for g in genes:
                key = (g['shift'], g['machine_id'])
                if key in used:
                    # Conflict! Try to repair
                    repaired = False
                    # First try different machine in same shift
                    for m in context['machine_ids']:
                        if (g['shift'], m) not in used:
                            g['machine_id'] = m
                            used.add((g['shift'], m))
                            repaired = True
                            break
                    
                    if not repaired:
                        # Try different shift & machine
                        for s in context['shifts']:
                            for m in context['machine_ids']:
                                if (s, m) not in used:
                                    g['shift'] = s
                                    g['machine_id'] = m
                                    used.add((s, m))
                                    repaired = True
                                    break
                            if repaired: break
                else:
                    used.add(key)

    @classmethod
    @transaction.atomic
    def _save_chromosome(cls, chromosome, fitness, context, department_id, week_start, user, generations, elapsed, seed=None, fitness_history=None):
        week_end = week_start + timedelta(days=6)
        
        algorithm_metadata = {
            'engine': 'GA_V1',
            'population_size': GAConfig.POPULATION_SIZE,
            'max_generations': GAConfig.MAX_GENERATIONS,
            'generations_run': generations,
            'elapsed_time_sec': round(elapsed, 4),
            'best_fitness': round(fitness, 2),
            'seed': seed,
            'fitness_history': fitness_history or [],
        }

        plan = SchedulePlan.objects.create(
            name=f"Lịch đề xuất GA - Tuần {week_start.strftime('%d/%m/%Y')}",
            department_id=department_id,
            week_start=week_start,
            week_end=week_end,
            status=SchedulePlan.Status.PROPOSED,
            fitness_score=fitness,
            algorithm_metadata=algorithm_metadata,
            created_by=user
        )
        
        assignments = []
        for g in chromosome:
            if 'shift_intervals' in context and (g['date'], g['shift']) in context['shift_intervals']:
                start_dt, end_dt = context['shift_intervals'][(g['date'], g['shift'])]
            else:
                start_dt, end_dt = cls._get_shift_interval(g['date'], g['shift'])
            
            assignments.append(ScheduleAssignment(
                schedule_plan=plan,
                patient_id=g['patient_id'],
                machine_id=g['machine_id'],
                scheduled_date=g['date'],
                shift=g['shift'],
                start_datetime=start_dt,
                end_datetime=end_dt,
                source=ScheduleAssignment.Source.GA
            ))
            
        ScheduleAssignment.objects.bulk_create(assignments)
        return plan
