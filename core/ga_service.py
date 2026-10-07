import math
import random
import statistics
import time as ptime
from datetime import timedelta, time, datetime
from collections import defaultdict
from django.db.models import Q
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
    def generate_plan(cls, department_id, week_start_date, user):
        start_time = ptime.time()
        
        # 1. Prepare Context
        context = cls._build_context(department_id, week_start_date)
        if not context['slots']:
            return None, "Không tìm thấy bệnh nhân hợp lệ cần phân ca trong khoa."
        if not context['machines']:
            return None, "Không tìm thấy máy lọc máu nào đang hoạt động."

        # 2. Init Population
        population = cls._init_population(context)
        
        best_chromosome = None
        best_fitness = float('inf')
        generations_no_improve = 0
        total_generations = 0
        
        # 3. Evolution Loop
        for gen in range(GAConfig.MAX_GENERATIONS):
            total_generations += 1
            # Evaluate Fitness
            fitnesses = [cls._evaluate_fitness(c, context) for c in population]
            
            # Find Best
            min_fit_idx = fitnesses.index(min(fitnesses))
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
                p1 = cls._tournament_selection(population, fitnesses)
                p2 = cls._tournament_selection(population, fitnesses)
                
                # Crossover
                if random.random() < GAConfig.CROSSOVER_PROB:
                    c1, c2 = cls._crossover(p1, p2, context)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Mutation
                cls._mutate(c1, context)
                cls._mutate(c2, context)
                
                # Validate & Repair
                cls._repair(c1, context)
                cls._repair(c2, context)
                
                new_population.append(c1)
                if len(new_population) < GAConfig.POPULATION_SIZE:
                    new_population.append(c2)
                    
            population = new_population

        elapsed = ptime.time() - start_time
        
        # 4. Check for Hard Violations
        if best_fitness >= GAConfig.HARD_PENALTY:
            return None, "Không thể tạo lịch hợp lệ với các máy và ràng buộc hiện tại."

        # 5. Save to DB
        plan = cls._save_chromosome(best_chromosome, best_fitness, context, department_id, week_start_date, user, total_generations, elapsed)
        return plan, None

    @classmethod
    def _build_context(cls, department_id, week_start):
        week_end = week_start + timedelta(days=6)
        
        patients = list(Patient.objects.filter(
            status__in=[Patient.Status.ACTIVE, Patient.Status.IN_TREATMENT, Patient.Status.STABLE],
            treatment_pattern__isnull=False
        ))
        
        machines = list(DialysisMachine.objects.filter(
            department_id=department_id
        ).exclude(status__in=[DialysisMachine.Status.BROKEN, DialysisMachine.Status.MAINTENANCE]))
        
        machine_ids = [m.machine_id for m in machines]
        
        existing_sessions = list(DialysisSession.objects.filter(
            machine__department_id=department_id,
            scheduled_start__date__gte=week_start,
            scheduled_start__date__lte=week_end,
            status__in=[DialysisSession.Status.SCHEDULED, DialysisSession.Status.IN_PROGRESS, DialysisSession.Status.COMPLETED]
        ))
        
        # historical machine for stability
        patient_history = {}
        for p in patients:
            last_sess = DialysisSession.objects.filter(
                patient=p,
                scheduled_start__date__lt=week_start
            ).order_by('-scheduled_start').first()
            if last_sess:
                patient_history[p.patient_id] = last_sess.machine_id

        slots = []
        patient_info = {}
        shifts = [Shift.SHIFT_1, Shift.SHIFT_2, Shift.SHIFT_3]
        
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
    def _init_population(cls, context):
        population = []
        for _ in range(GAConfig.POPULATION_SIZE):
            chromosome = []
            
            # Group slots by date to avoid machine duplication heuristically
            slots_by_date = defaultdict(list)
            for slot in context['slots']:
                slots_by_date[slot['date']].append(slot)
                
            for date, daily_slots in slots_by_date.items():
                # Randomize order
                random.shuffle(daily_slots)
                
                # Keep track of used (shift, machine) on this date
                used = set()
                
                for slot in daily_slots:
                    shift = slot['preferred_shift'] if slot['preferred_shift'] and random.random() > 0.3 else random.choice(context['shifts'])
                    
                    available_machines = [m for m in context['machine_ids'] if (shift, m) not in used]
                    
                    if not available_machines:
                        # Fallback: pick any shift that has available machines
                        for s in context['shifts']:
                            available_machines = [m for m in context['machine_ids'] if (s, m) not in used]
                            if available_machines:
                                shift = s
                                break
                    
                    if available_machines:
                        machine_id = random.choice(available_machines)
                        used.add((shift, machine_id))
                    else:
                        # Force assign (will be penalized later)
                        machine_id = random.choice(context['machine_ids'])
                        
                    chromosome.append({
                        'patient_id': slot['patient_id'],
                        'date': date,
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
        
        for gene in chromosome:
            # 1. Hard: duplicate machine/shift/date
            key = (gene['date'], gene['shift'], gene['machine_id'])
            if key in used_slots:
                penalty += GAConfig.HARD_PENALTY
            used_slots.add(key)
            
            # 2. Hard: overlap with existing session
            for sess in context['existing_sessions']:
                if sess.machine_id == gene['machine_id'] and sess.scheduled_start.date() == gene['date']:
                    # Check time overlap
                    gene_start, gene_end = SHIFT_TIMINGS[gene['shift']]
                    sess_start = sess.scheduled_start.time()
                    sess_end = sess.scheduled_end.time()
                    if max(gene_start, sess_start) < min(gene_end, sess_end):
                        penalty += GAConfig.HARD_PENALTY
                        break
                        
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
    def _tournament_selection(cls, population, fitnesses):
        best = None
        best_fit = float('inf')
        for _ in range(GAConfig.TOURNAMENT_SIZE):
            idx = random.randint(0, len(population) - 1)
            if fitnesses[idx] < best_fit:
                best_fit = fitnesses[idx]
                best = population[idx]
        return best

    @classmethod
    def _crossover(cls, p1, p2, context):
        # Day-based crossover
        c1, c2 = [], []
        # Group by date
        p1_by_date = defaultdict(list)
        p2_by_date = defaultdict(list)
        for g in p1: p1_by_date[g['date']].append(g)
        for g in p2: p2_by_date[g['date']].append(g)
        
        dates = list(p1_by_date.keys())
        for d in dates:
            if random.random() < 0.5:
                c1.extend([dict(g) for g in p1_by_date[d]])
                c2.extend([dict(g) for g in p2_by_date[d]])
            else:
                c1.extend([dict(g) for g in p2_by_date[d]])
                c2.extend([dict(g) for g in p1_by_date[d]])
                
        return c1, c2

    @classmethod
    def _mutate(cls, chromosome, context):
        if random.random() > GAConfig.MUTATION_PROB:
            return
            
        # Swap mutation within same date
        dates = list(set(g['date'] for g in chromosome))
        target_date = random.choice(dates)
        genes_in_date = [g for g in chromosome if g['date'] == target_date]
        if len(genes_in_date) >= 2:
            g1, g2 = random.sample(genes_in_date, 2)
            # Swap (shift, machine_id)
            temp_s, temp_m = g1['shift'], g1['machine_id']
            g1['shift'], g1['machine_id'] = g2['shift'], g2['machine_id']
            g2['shift'], g2['machine_id'] = temp_s, temp_m

    @classmethod
    def _repair(cls, chromosome, context):
        # Check and fix hard constraints within chromosome (duplicate machine per shift)
        by_date = defaultdict(list)
        for g in chromosome:
            by_date[g['date']].append(g)
            
        for date, genes in by_date.items():
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
    def _save_chromosome(cls, chromosome, fitness, context, department_id, week_start, user, generations, elapsed):
        week_end = week_start + timedelta(days=6)
        
        plan = SchedulePlan.objects.create(
            name=f"Lịch đề xuất GA - Tuần {week_start.strftime('%d/%m/%Y')}",
            department_id=department_id,
            week_start=week_start,
            week_end=week_end,
            status=SchedulePlan.Status.PROPOSED,
            fitness_score=fitness,
            algorithm_metadata={
                'population_size': GAConfig.POPULATION_SIZE,
                'max_generations': GAConfig.MAX_GENERATIONS,
                'generations_run': generations,
                'elapsed_time_sec': round(elapsed, 2),
                'best_fitness': round(fitness, 2)
            },
            created_by=user
        )
        
        assignments = []
        for g in chromosome:
            start_time, end_time = SHIFT_TIMINGS[g['shift']]
            start_dt = datetime.combine(g['date'], start_time)
            end_dt = datetime.combine(g['date'], end_time)
            
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
