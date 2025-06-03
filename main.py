from __future__ import annotations
import json
import sys
import pathlib
import random
from typing import Dict, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import mesa


DEFAULTS: Dict[str, Any] = {
    "GRID_WIDTH": 300,
    "GRID_HEIGHT": 300,
    "STEPS_PER_YEAR": 52,
    "SIM_YEARS": 3,
    "INITIAL_POP_DEFAULT": 50,
    "ANNUAL_MAINLAND_IMMIGRATION_RATE": 0.005,
}

# GIS
GALAPAGOS_GIS_FILE = "gadm41_ECU_2.json"
ISLAND_NAME_COLUMN_IN_GIS = "NAME_2"

# Habitat
HABITAT_TYPES = {"Sea": 0, "Coastal": 1, "Scrubland": 2, "Highland": 3}
COASTAL_ZONE_DEPTH = 3
HIGHLAND_THRESHOLD_AREA = 50
HIGHLAND_CORE_RATIO = 0.3

# species json
SPECIES_JSON_PATH = pathlib.Path(__file__).with_name("species_data.json")


def load_species_json(path: pathlib.Path = SPECIES_JSON_PATH) -> Dict[str, Any]:
    """Read species definitions from *.json* and massage tuples."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for s in raw.values():
            if isinstance(s.get("initial_energy_range"), list):
                s["initial_energy_range"] = tuple(s["initial_energy_range"])
        return raw
    except Exception as e:
        print(f"[FATAL] cannot load species json: {e}", file=sys.stderr)
        raise


SPECIES_DATA_ANNUAL = load_species_json()


def normalize_simulation_data(annual_species_data: Dict[str, Any],
                              annual_immigration_rate: float,
                              steps_per_year: int):
    """Convert annual rates/durations to *per-step* values."""
    processed_species_data: Dict[str, Any] = {}
    for sp, cfg in annual_species_data.items():
        c = cfg.copy()
        c["repro_rate"] = c["annual_repro_rate"] / steps_per_year
        c["mortality_base"] = c["annual_mortality_base"] / steps_per_year
        c["max_age"] = c["max_age_years"] * steps_per_year
        for k in ("annual_repro_rate", "annual_mortality_base", "max_age_years"):
            c.pop(k, None)
        processed_species_data[sp] = c

    return processed_species_data, annual_immigration_rate / steps_per_year


def get_habitat_name(hab_id: int) -> str:
    return next((k for k, v in HABITAT_TYPES.items() if v == hab_id), "Unknown")


class BirdAgent(mesa.Agent):

    def __init__(self, uid: int, model: "GalapagosModel",
                 species_name: str, initial_pos: Tuple[int, int]):
        super().__init__(uid, model)
        self.species_name = species_name
        self.s_data = model.species_data_processed[species_name]

        self.age = 0
        self.energy = random.randint(*self.s_data["initial_energy_range"])

        if 0 <= initial_pos[0] < model.grid.width and 0 <= initial_pos[1] < model.grid.height:
            self.current_island_id = model.island_map_ids[initial_pos]
            self.current_habitat_id = model.habitat_map[initial_pos]
        else:
            self.current_island_id = -1
            self.current_habitat_id = HABITAT_TYPES["Sea"]

    def get_preferred_moves(self, nbrs):
        scored = []
        for mv in nbrs:
            hid = self.model.habitat_map[mv]
            hname = get_habitat_name(hid)
            score = self.s_data["habitat_preference_score"].get(hname, 0)

            if hname == "Sea" and getattr(self, "target_island_direction", None):
                dx, dy = mv[0]-self.pos[0], mv[1]-self.pos[1]
                if (dx * self.target_island_direction[0] +
                        dy * self.target_island_direction[1]) > 0:
                    score += 2
            scored.append((score, mv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def sense_islands(self):
        self.target_island_direction = None
        if self.model.habitat_map[self.pos] != HABITAT_TYPES["Sea"]:
            return
        best = (float("inf"), None)
        for _ in range(8):
            angle = random.uniform(0, 2*np.pi)
            for r in range(1, int(self.s_data["sensing_range"])):
                cx = int(self.pos[0] + r*np.cos(angle))
                cy = int(self.pos[1] + r*np.sin(angle))
                if not (0 <= cx < self.model.grid.width and 0 <= cy < self.model.grid.height):
                    break
                if self.model.island_map_types[cx, cy] == "land":
                    dist = np.hypot(cx-self.pos[0], cy-self.pos[1])
                    if dist < best[0]:
                        best = (dist, (cx, cy))
                    break
        if best[1]:
            dx, dy = best[1][0]-self.pos[0], best[1][1]-self.pos[1]
            norm = np.hypot(dx, dy)
            self.target_island_direction = (dx/norm, dy/norm)

    def move(self):
        current_name = get_habitat_name(self.model.habitat_map[self.pos])
        nbrs = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        if not nbrs:
            return

        if current_name == "Sea":
            self.sense_islands()

        pref = self.get_preferred_moves(nbrs)
        choice = random.choice(pref) if not pref else pref[0]

        dest_name = get_habitat_name(self.model.habitat_map[choice])
        cost = self.s_data["move_cost_water"] if dest_name == "Sea" else self.s_data["move_cost_land"]

        if self.energy > cost:
            self.energy -= cost
            self.model.grid.move_agent(self, choice)
        else:
            self.energy -= self.s_data["move_cost_land"]

        self.current_island_id = self.model.island_map_ids[self.pos]
        self.current_habitat_id = self.model.habitat_map[self.pos]

    def gain_energy(self):
        hname = get_habitat_name(self.model.habitat_map[self.pos])
        regen = self.s_data["energy_regen_factor"].get(hname, 0)
        base = (self.s_data["base_energy_regen_land"]
                if hname != "Sea" else self.s_data.get("base_energy_regen_sea", 0))
        self.energy = min(self.s_data["max_energy"], self.energy + base*regen)

    def check_mortality(self):
        if self.energy <= 0 or self.age > self.s_data["max_age"]:
            return True
        hname = get_habitat_name(self.model.habitat_map[self.pos])
        ps = self.s_data["habitat_preference_score"].get(hname, 0)
        mod = 2.0 if hname == "Sea" else (1.5 if ps < 3 else 1.0)
        return random.random() < self.s_data["mortality_base"]*mod

    def reproduce(self):
        hname = get_habitat_name(self.model.habitat_map[self.pos])
        if hname == "Sea":
            return
        bonus = self.s_data["repro_habitat_bonus"].get(hname, 1.0)
        if (self.energy >= self.s_data["min_repro_energy"] and
                random.random() < self.s_data["repro_rate"]*bonus):
            self.energy -= self.s_data["repro_energy_cost"]
            nbrs = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=True)
            land = [p for p in nbrs if self.model.habitat_map[p]
                    != HABITAT_TYPES["Sea"]]
            spawn = random.choice(land) if land else self.pos
            baby = BirdAgent(self.model.next_agent_id(),
                             self.model, self.species_name, spawn)
            self.model.grid.place_agent(baby, spawn)
            self.model.schedule.add(baby)

    def step(self):
        self.age += 1
        self.move()
        self.gain_energy()
        if self.check_mortality():
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
        else:
            self.reproduce()


class GalapagosModel(mesa.Model):
    """Mesa model; now supports custom per-island population."""

    def __init__(self,
                 width: int, height: int,
                 gis_file: str, island_name_col: str,
                 processed_species_data: Dict[str, Any],
                 default_init_pop: int,
                 mainland_immigration_rate_per_step: float,
                 mainland_point: Tuple[int, int],
                 initial_pop_distribution: Optional[Dict[str, Dict[int, int]]] = None):
        super().__init__()
        self.width, self.height = width, height
        self.gis_file, self.island_name_col = gis_file, island_name_col
        self.species_data_processed = processed_species_data
        self.initial_pop_default = default_init_pop
        self.mainland_immigration_rate_per_step = mainland_immigration_rate_per_step
        self.mainland_point = mainland_point
        self.init_pop_dist = initial_pop_distribution or {}
        self._agent_id_counter = 0

        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)

        self.island_map_types = np.full((width, height), "sea", dtype=object)
        self.island_map_ids = np.full((width, height), -1,   dtype=int)
        self.habitat_map = np.full(
            (width, height), HABITAT_TYPES["Sea"], dtype=int)
        self.islands_info: list[dict] = []

        self._load_and_process_gis()
        self._assign_habitats()
        self._initialize_agents()
        self._setup_datacollector()

        self.running = True
        self.datacollector.collect(self)

    # util
    def next_agent_id(self):
        self._agent_id_counter += 1
        return self._agent_id_counter

    # GIS
    def _load_and_process_gis(self):
        try:
            full = gpd.read_file(self.gis_file)
        except Exception as e:
            print(f"[GIS] ERROR reading {self.gis_file}: {e}")
            return

        gdf = full[full["NAME_1"] ==
                   "Galápagos"] if "NAME_1" in full.columns else full
        minx, miny, maxx, maxy = gdf.total_bounds
        self.x_scale = self.width / (maxx-minx) if maxx != minx else 1
        self.y_scale = self.height/(maxy-miny) if maxy != miny else 1
        self.x_offset, self.y_offset = minx, miny

        for idx, row in gdf.iterrows():
            geom, name = row.geometry, row.get(
                self.island_name_col, f"Isl_{idx}")
            cells = []
            for gx in range(self.width):
                for gy in range(self.height):
                    ux = (gx+0.5)/self.x_scale + self.x_offset
                    uy = (gy+0.5)/self.y_scale + self.y_offset
                    if geom.contains(Point(ux, uy)):
                        self.island_map_types[gx, gy] = "land"
                        self.island_map_ids[gx, gy] = idx
                        cells.append((gx, gy))
            if not cells:
                continue
            cent = geom.centroid
            cgx = int((cent.x-self.x_offset)*self.x_scale)
            cgy = int((cent.y-self.y_offset)*self.y_scale)
            self.islands_info.append(
                {"id": idx, "name": name, "geometry_orig": geom,
                 "area_cells": len(cells), "centroid_grid": (cgx, cgy),
                 "cells_coords": cells, "coastal_cells": []}
            )

    def _assign_habitats(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.island_map_types[x, y] != "land":
                    continue
                # detect coastal
                coastal = False
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == dy == 0:
                            continue
                        for d in range(1, COASTAL_ZONE_DEPTH+1):
                            nx, ny = x+dx*d, y+dy*d
                            if not (0 <= nx < self.width and 0 <= ny < self.height):
                                coastal = True
                                break
                            if self.island_map_types[nx, ny] == "sea":
                                coastal = True
                                break
                        if coastal:
                            break
                    if coastal:
                        break
                if coastal:
                    self.habitat_map[x, y] = HABITAT_TYPES["Coastal"]
                    self._mark_coastal_cell(x, y)
                else:
                    isl_id = self.island_map_ids[x, y]
                    isl = next(
                        i for i in self.islands_info if i["id"] == isl_id)
                    if isl["area_cells"] > HIGHLAND_THRESHOLD_AREA:
                        dist = np.hypot(
                            x-isl["centroid_grid"][0], y-isl["centroid_grid"][1])
                        radius = np.sqrt(isl["area_cells"]/np.pi)
                        self.habitat_map[x, y] = (HABITAT_TYPES["Highland"]
                                                  if dist < radius*HIGHLAND_CORE_RATIO else HABITAT_TYPES["Scrubland"])
                    else:
                        self.habitat_map[x, y] = HABITAT_TYPES["Scrubland"]

    def _mark_coastal_cell(self, x, y):
        isl_id = self.island_map_ids[x, y]
        for isl in self.islands_info:
            if isl["id"] == isl_id:
                isl["coastal_cells"].append((x, y))
                break

    def _initialize_agents(self):
        for isl in self.islands_info:
            land_cells = [c for c in isl["cells_coords"]
                          if self.habitat_map[c] != HABITAT_TYPES["Sea"]]
            for sp in self.species_data_processed:
                n = self.init_pop_dist.get(sp, {}).get(
                    isl["id"], self.initial_pop_default)
                for _ in range(n):
                    pos = random.choice(land_cells)
                    a = BirdAgent(self.next_agent_id(), self, sp, pos)
                    self.grid.place_agent(a, pos)
                    self.schedule.add(a)

    def _setup_datacollector(self):
        reporters = {"TotalAgents": lambda m: m.schedule.get_agent_count()}
        for sp in self.species_data_processed:
            reporters[f"{sp}_total"] = lambda m, s=sp: sum(
                1 for a in m.schedule.agents if a.species_name == s)
        for isl in self.islands_info:
            rid = isl["id"]
            tag = isl["name"].replace(" ", "_")
            reporters[f"Richness_{tag}"] = lambda m, rid=rid: len({a.species_name for a in m.schedule.agents
                                                                  if getattr(a, "current_island_id", -99) == rid})
        self.datacollector = mesa.DataCollector(model_reporters=reporters)

    def apply_mainland_immigration(self):
        for isl in self.islands_info:
            cells = isl["coastal_cells"] or isl["cells_coords"]
            if not cells:
                continue
            factor = len(cells)/50.0
            for sp in self.species_data_processed:
                if random.random() < self.mainland_immigration_rate_per_step*factor:
                    pos = random.choice(cells)
                    a = BirdAgent(self.next_agent_id(), self, sp, pos)
                    self.grid.place_agent(a, pos)
                    self.schedule.add(a)

    def step(self):
        if self.schedule.get_agent_count():
            self.apply_mainland_immigration()
            self.schedule.step()
        self.datacollector.collect(self)


def plot_species_area_relationship(model_data_df, islands_info_list):
    if not islands_info_list:
        print("SAR: No island info.")
        return
    richness, areas, names = [], [], []
    for isl in islands_info_list:
        col = f"Richness_{isl['name'].replace(' ', '_')}"
        if col in model_data_df.columns:
            avg_r = model_data_df[col].tail(min(50, len(model_data_df))).mean()
            richness.append(avg_r)
            areas.append(isl["area_cells"])
            names.append(isl["name"])
        else:
            print(f"SAR: Column {col} not found.")
    if not richness or not areas:
        print("SAR: Not enough data.")
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(areas, richness, color='blue')
    for i, name in enumerate(names):
        plt.annotate(name, (areas[i], richness[i]), textcoords="offset points", xytext=(
            0, 5), ha='center', fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Island Area (Cells) - Log")
    plt.ylabel(
        f"Species Richness (Avg. Last {min(50,len(model_data_df))} Weeks) - Log")
    plt.title("Species-Area Relationship (SAR)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_species_isolation_relationship(model_data_df, islands_info_list, mainland_point):
    if not islands_info_list:
        print("SIR: No island info.")
        return
    richness, distances, names = [], [], []
    for isl in islands_info_list:
        col = f"Richness_{isl['name'].replace(' ', '_')}"
        if col in model_data_df.columns:
            avg_r = model_data_df[col].tail(min(50, len(model_data_df))).mean()
            dist = np.sqrt((isl["centroid_grid"][0]-mainland_point[0])
                           ** 2 + (isl["centroid_grid"][1]-mainland_point[1])**2)
            richness.append(avg_r)
            distances.append(dist)
            names.append(isl["name"])
        else:
            print(f"SIR: Column {col} not found.")
    if not richness or not distances:
        print("SIR: Not enough data.")
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, richness, color='green')
    for i, name in enumerate(names):
        plt.annotate(name, (distances[i], richness[i]), textcoords="offset points", xytext=(
            0, 5), ha='center', fontsize=8)
    plt.xlabel("Isolation (Distance to Mainland Point)")
    plt.ylabel(
        f"Species Richness (Avg. Last {min(50,len(model_data_df))} Weeks)")
    plt.title("Species-Isolation Relationship (SIR)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_final_distribution(model):
    plt.figure(figsize=(12, 12 * (model.height /
               model.width if model.width else 1)))
    colors = {
        HABITAT_TYPES["Sea"]: [0.7, 0.85, 1.0],
        HABITAT_TYPES["Coastal"]: [0.9, 0.85, 0.7],
        HABITAT_TYPES["Scrubland"]: [0.6, 0.8, 0.6],
        HABITAT_TYPES["Highland"]: [0.4, 0.6, 0.4]
    }
    rgb_map = np.zeros((model.height, model.width, 3))
    for r in range(model.height):
        for c in range(model.width):
            rgb_map[r, c, :] = colors.get(model.habitat_map[c, r], [0, 0, 0])
    plt.imshow(rgb_map, origin="lower", extent=[
               0, model.width, 0, model.height])

    species_set = {agent.species_name for agent in model.schedule.agents if isinstance(
        agent, BirdAgent)}
    species_list = sorted(species_set)

    if not species_list:
        plt.title(
            f"Agent Distribution at Step {model.schedule.steps} (tidak ada agen)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, ls=':', alpha=0.2)
        plt.xlim(0, model.width)
        plt.ylim(0, model.height)
        plt.tight_layout()
        plt.show()
        return

    palette = plt.colormaps['tab10'].resampled(len(species_list))
    s_colors = {sp: palette(i) for i, sp in enumerate(species_list)}

    ax, ay, ac = [], [], []
    for agent in model.schedule.agents:
        if isinstance(agent, BirdAgent) and agent.pos:
            ax.append(agent.pos[0] + 0.5)
            ay.append(agent.pos[1] + 0.5)
            ac.append(s_colors.get(agent.species_name, "black"))
    if ax:
        plt.scatter(ax, ay, c=ac, s=10, marker="o", alpha=0.7,
                    edgecolor='gray', linewidth=0.3)

    handles = []
    labels = []
    for sp, col in s_colors.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=col, markersize=6, alpha=0.7,
                                  markeredgecolor='gray', markeredgewidth=0.3))
        labels.append(sp)
    plt.legend(handles, labels, title="Species", loc="upper right", fontsize=8)

    if hasattr(model, 'islands_info'):
        for isl in model.islands_info:
            if isl["area_cells"] > 0:
                plt.text(
                    isl["centroid_grid"][0], isl["centroid_grid"][1], isl["name"],
                    ha='center', va='center', color='black', fontsize=7,
                    bbox=dict(facecolor='white', alpha=0.5, pad=0.2,
                              boxstyle='round,pad=0.1')
                )

    plt.title(f"Agent Distribution at Step {model.schedule.steps}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, ls=':', alpha=0.2)
    plt.xlim(0, model.width)
    plt.ylim(0, model.height)
    plt.tight_layout()
    plt.show()


def run_simulation(settings: Dict[str, Any],
                   species_dict: Dict[str, Any],
                   initial_pop_distribution: Optional[Dict[str, Dict[int, int]]] = None):
    """Utility for scripts / UI."""
    settings = {**DEFAULTS, **settings}
    steps_per_year = settings["STEPS_PER_YEAR"]
    sim_steps = steps_per_year*settings["SIM_YEARS"]

    proc_sp, imm_per_step = normalize_simulation_data(
        species_dict,
        settings["ANNUAL_MAINLAND_IMMIGRATION_RATE"],
        steps_per_year
    )
    mainland_pt = (settings["GRID_WIDTH"]+20, settings["GRID_HEIGHT"]//2)

    model = GalapagosModel(
        width=settings["GRID_WIDTH"],
        height=settings["GRID_HEIGHT"],
        gis_file=GALAPAGOS_GIS_FILE,
        island_name_col=ISLAND_NAME_COLUMN_IN_GIS,
        processed_species_data=proc_sp,
        default_init_pop=settings["INITIAL_POP_DEFAULT"],
        mainland_immigration_rate_per_step=imm_per_step,
        mainland_point=mainland_pt,
        initial_pop_distribution=initial_pop_distribution
    )

    for i in range(sim_steps):
        print(f"Running Step-{i}")
        model.step()
        if model.schedule.get_agent_count() == 0:
            break

    df = model.datacollector.get_model_vars_dataframe()
    return model, df


if __name__ == "__main__":
    print("Running with defaults …")
    m, df = run_simulation(DEFAULTS, SPECIES_DATA_ANNUAL)
    print(df.tail())
    from matplotlib import pyplot as plt
    for sp in SPECIES_DATA_ANNUAL:
        col = f"{sp}_total"
        if col in df.columns:
            plt.plot(df.index, df[col], label=sp)
    plt.legend()
    plt.xlabel("week")
    plt.ylabel("pop")
    plt.show()
