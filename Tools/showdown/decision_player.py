"""Pokémon Showdown player driven by a typed-decision model (Intern-Decision-0.8B on Core ML).

Every turn the battle is rendered as one request in Intern-Decision's wire format: a JSON state (our team, the
opponent's revealed team, field) and one `choice` question whose options are the legal moves and switches with their
facts (type, category, power, accuracy, PP, effectiveness against the opponent's active Pokémon). The model answers
with a probability per option; the top option is played. Every call is logged (state, options, probabilities,
latency) so the same log can label a student or replay a decision.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from poke_env.battle import Battle, Move, Pokemon
from poke_env.player import Player

COREML_DIR = Path(__file__).resolve().parent / "coreml"
sys.path.insert(0, str(COREML_DIR))
from decision_export import Compiler, field_probabilities, row_inputs  # noqa: E402

QUESTION = "Which action should the player take this turn to win the battle?"


class CoreMLDecider:
    """Intern-Decision-0.8B `DecisionRow` buckets (`L<len>_F<fields>/`) + `embeddings.f16`, as published on the Hub."""

    def __init__(self, model_dir: Path, checkpoint: Path, compute_units: str = "cpu_gpu"):
        import coremltools as ct

        units = {"all": ct.ComputeUnit.ALL, "cpu_gpu": ct.ComputeUnit.CPU_AND_GPU, "cpu": ct.ComputeUnit.CPU_ONLY}
        self.compiler = Compiler(checkpoint)
        self.buckets = []  # (length, max_fields, function_name or None, package)
        config = None
        top = model_dir / "config.json"
        if top.exists() and "functions" in json.loads(top.read_text()):
            # one weight-shared multifunction package: one function per bucket
            config = json.loads(top.read_text())
            package = next(model_dir.glob("*.mlpackage"))
            for name, spec in config["functions"].items():
                self.buckets.append((spec["length"], spec["max_fields"], name, package))
        else:
            for folder in sorted(model_dir.glob("L*_F*")):
                config = json.loads((folder / "config.json").read_text())
                package = next(iter(folder.glob("DecisionRow_fp16.mlpackage")), None) or next(folder.glob("*.mlpackage"))
                self.buckets.append((config["length"], config["max_fields"], None, package))
        self.buckets.sort(key=lambda b: b[0])
        self.models = {}
        self.units = units[compute_units]
        self.config = config
        self.cfg = SimpleNamespace(rotary_dim=config["rotary_dim"], rope_theta=config["rope_theta"],
                                   mrope_section=[11, 11, 10])
        table = np.fromfile(model_dir / "embeddings.f16", dtype=np.float16)
        self.embed = torch.from_numpy(table.reshape(config["vocab_size"], config["hidden_size"]).astype(np.float32))
        self.temperature = config["temperature"]
        self.pad_id = config["pad_id"]

    def model(self, package: Path, function_name: str | None = None):
        import coremltools as ct

        key = (package, function_name)
        if key not in self.models:
            kwargs = {"function_name": function_name} if function_name else {}
            self.models[key] = ct.models.MLModel(str(package), compute_units=self.units, **kwargs)
        return self.models[key]

    def warm(self):
        """Load every bucket and run one prediction each: the first call of a bucket pays Core ML's specialization."""
        for length, max_fields, function_name, package in self.buckets:
            model = self.model(package, function_name)
            inputs = row_inputs(self.cfg, self.embed, [self.pad_id] * 8, [3], length, max_fields, self.pad_id)
            model.predict({n: t.numpy().astype(np.float32) for n, t in zip(["hidden", "cos", "sin", "field_onehot"], inputs)})

    def decide(self, request: dict) -> dict:
        ids, positions, fields, _ = self.compiler.encode(request)
        for length, max_fields, function_name, package in self.buckets:
            if len(ids) <= length and len(positions) <= max_fields:
                break
        else:
            raise ValueError(f"request needs {len(ids)} tokens; largest bucket is {self.buckets[-1][0]}")
        inputs = row_inputs(self.cfg, self.embed, ids, positions, length, max_fields, self.pad_id)
        feed = {n: t.numpy().astype(np.float32) for n, t in zip(["hidden", "cos", "sin", "field_onehot"], inputs)}
        start = time.perf_counter()
        logits = torch.from_numpy(np.asarray(self.model(package, function_name).predict(feed)["logits"]))
        ms = (time.perf_counter() - start) * 1000
        answers = {}
        for (name, labels), (_, scaled) in zip(fields, field_probabilities(logits, fields, self.temperature)):
            answers[name] = dict(zip(labels, scaled.tolist()))
        return {"answers": answers, "tokens": len(ids), "bucket": length, "ms": ms}


class HFDecider:
    """Any Intern-Decision checkpoint through its own `inference.py` (PyTorch, MPS): the reference path, for
    comparing sizes without a Core ML export."""

    def __init__(self, checkpoint: Path, dtype: str = "bfloat16"):
        import importlib.util

        spec = importlib.util.spec_from_file_location("intern_decision_inference_hf", checkpoint / "inference.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses in the script look their module up by name
        spec.loader.exec_module(module)
        self.engine = module.DecisionEngine(str(checkpoint), device="mps", dtype=dtype)

    def warm(self):
        pass

    def decide(self, request: dict) -> dict:
        result = self.engine.predict(request)
        answers = {name: answer["probabilities"] for name, answer in result["answers"].items()}
        return {"answers": answers, "tokens": result["usage"]["input_tokens"], "bucket": 0,
                "ms": result["timing"]["inference_ms"]}


class HeuristicDecider:
    """Reads the same request text: strongest move by power x effectiveness, else the switch with the best matchup.
    Validates the option -> order mapping and gives a floor for the model."""

    def decide(self, request: dict) -> dict:
        import re

        criteria = request["questions"]["action"]["criteria"]
        scores = {}
        for label, text in criteria.items():
            if label.startswith("use "):
                power = re.search(r"power (\d+)", text)
                eff = re.search(r"(\d*\.?\d+)x", text)
                stab = 1.5 if "STAB" in text else 1.0
                scores[label] = 1 + (int(power.group(1)) if power else 0) * (float(eff.group(1)) if eff else 1) * stab
            else:
                takes = re.search(r"takes (\d*\.?\d+)x", text)
                deals = re.search(r"deals (\d*\.?\d+)x", text)
                scores[label] = 0.5 * (float(deals.group(1)) if deals else 1) / max(0.25, float(takes.group(1)) if takes else 1)
        total = sum(scores.values())
        return {"answers": {"action": {k: v / total for k, v in scores.items()}}, "tokens": 0, "bucket": 0, "ms": 0.0}


def hp(pokemon: Pokemon) -> str:
    return f"{round(pokemon.current_hp_fraction * 100)}%"


def types(pokemon: Pokemon) -> str:
    return "/".join(t.name.capitalize() for t in pokemon.types if t)


def boosts(pokemon: Pokemon) -> dict:
    return {k: v for k, v in pokemon.boosts.items() if v}


def effectiveness(move: Move, target: Pokemon | None) -> float:
    if target is None or move.category.name == "STATUS":
        return 1.0
    return target.damage_multiplier(move)


def line(pokemon: Pokemon, ours: bool, active: bool = True) -> str:
    parts = [pokemon.species]
    if active or not ours:
        parts.append(types(pokemon))
    parts.append(hp(pokemon))
    if pokemon.status:
        parts.append(pokemon.status.name.lower())
    if active and boosts(pokemon):
        parts.append(" ".join(f"{k}{v:+d}" for k, v in boosts(pokemon).items()))
    if not ours and active and pokemon.moves:
        parts.append("seen: " + " ".join(pokemon.moves))
    return " ".join(parts)


def render_state(battle: Battle) -> dict:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    bench = [p for p in battle.team.values() if not p.active and not p.fainted]
    opp_bench = [p for p in battle.opponent_team.values() if not p.active and not p.fainted]
    unrevealed = max(0, (battle.max_team_size or 6) - len(battle.opponent_team))
    state = {
        "turn": battle.turn,
        "our_active": line(me, True) if me else None,
        "our_bench": [line(p, True, active=False) for p in bench],
        "opponent_active": line(opp, False) if opp else None,
        "opponent_bench": [line(p, False, active=False) for p in opp_bench]
        + ([f"{unrevealed} unrevealed"] if unrevealed else []),
    }
    extras = []
    if battle.weather:
        extras += [w.name.lower() for w in battle.weather]
    if battle.fields:
        extras += [f.name.lower() for f in battle.fields]
    if battle.side_conditions:
        extras.append("our side: " + " ".join(c.name.lower() for c in battle.side_conditions))
    if battle.opponent_side_conditions:
        extras.append("their side: " + " ".join(c.name.lower() for c in battle.opponent_side_conditions))
    if extras:
        state["field"] = extras
    return state


def attribute(move: Move, name: str, default):
    """poke-env raises on moves whose data entry lacks a field (e.g. no `priority`); treat those as the default."""
    try:
        value = getattr(move, name)
    except (KeyError, AttributeError, TypeError):
        return default
    return default if value is None else value


def move_option(move: Move, opp: Pokemon | None, me: Pokemon | None) -> str:
    category = attribute(move, "category", None)
    cat = category.name.lower() if category else "status"
    move_type = attribute(move, "type", None)
    parts = [f"{move_type.name.capitalize() if move_type else 'Unknown'} {cat}"]
    if cat != "status":
        stab = " STAB" if me and move_type and move_type in me.types else ""
        parts.append(f"power {attribute(move, 'base_power', 0)}{stab}")
        if opp:
            try:
                parts.append(f"{opp.damage_multiplier(move):g}x")
            except (KeyError, AttributeError, TypeError):
                pass
    acc = attribute(move, "accuracy", 1)
    parts.append("acc 100" if acc is True or acc == 1 else f"acc {round(float(acc) * 100)}")
    parts.append(f"PP {attribute(move, 'current_pp', 0)}")
    priority = attribute(move, "priority", 0)
    if priority:
        parts.append(f"priority {priority:+d}")
    heal = attribute(move, "heal", 0)
    if heal:
        parts.append(f"heals {round(heal * 100)}%")
    boosts_ = attribute(move, "boosts", None)
    if boosts_:
        parts.append("boosts " + " ".join(f"{k}{v:+d}" for k, v in boosts_.items()))
    self_boost = attribute(move, "self_boost", None)
    if self_boost:
        parts.append("self " + " ".join(f"{k}{v:+d}" for k, v in self_boost.items()))
    status = attribute(move, "status", None)
    if status:
        parts.append(f"inflicts {status.name.lower()}")
    return ", ".join(parts)


def switch_option(pokemon: Pokemon, opp: Pokemon | None) -> str:
    parts = [f"{types(pokemon)} {hp(pokemon)} HP"]
    if pokemon.status:
        parts.append(pokemon.status.name.lower())
    if opp:
        incoming = max((pokemon.damage_multiplier(t) for t in opp.types if t), default=1.0)
        outgoing = max((opp.damage_multiplier(t) for t in pokemon.types if t), default=1.0)
        parts.append(f"takes {incoming:g}x, deals {outgoing:g}x")
    return ", ".join(parts)


def render_options(battle: Battle) -> list[tuple[str, str, object]]:
    """(label, description, order-target) for every legal action."""
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon
    options = []
    if not battle.force_switch:
        for move in battle.available_moves:
            options.append((f"use {move.id}", move_option(move, opp, me), move))
    for pokemon in battle.available_switches:
        options.append((f"switch to {pokemon.species}", switch_option(pokemon, opp), pokemon))
    return options


class InternDecisionPlayer(Player):
    def __init__(self, decider, log_path: Path | None = None, permutations: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.decider = decider
        self.permutations = permutations
        self.rng = __import__("random").Random(0)
        self.log = open(log_path, "a") if log_path else None
        self.latencies = []
        self.tokens = []

    def choose_move(self, battle: Battle):
        options = render_options(battle)
        if not options:
            return self.choose_default_move()
        if len(options) == 1:
            return self.create_order(options[0][2])
        state = render_state(battle)
        probabilities = {label: 0.0 for label, _, _ in options}
        result = None
        for k in range(self.permutations):
            ordered = list(options)
            if k:
                self.rng.shuffle(ordered)
            request = {"state": state,
                       "questions": {"action": {"type": "choice", "instructions": QUESTION,
                                                "criteria": {label: text for label, text, _ in ordered}}}}
            try:
                result = self.decider.decide(request)
            except ValueError as error:
                self.logger.warning("request too long (%s); random move", error)
                return self.choose_random_move(battle)
            for label, p in result["answers"]["action"].items():
                probabilities[label] += p / self.permutations
        request = {"state": state, "questions": {"action": {"type": "choice", "instructions": QUESTION,
                                                             "criteria": {label: text for label, text, _ in options}}}}
        best = max(options, key=lambda option: probabilities[option[0]])
        self.latencies.append(result["ms"])
        self.tokens.append(result["tokens"])
        if self.log:
            self.log.write(json.dumps({"battle": battle.battle_tag, "turn": battle.turn, "request": request,
                                       "probabilities": probabilities, "chosen": best[0], "tokens": result["tokens"],
                                       "bucket": result["bucket"], "ms": result["ms"]}) + "\n")
            self.log.flush()
        return self.create_order(best[2])
