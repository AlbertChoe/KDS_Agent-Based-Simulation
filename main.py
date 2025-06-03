import mesa
import random
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# --- Configuration ---
GRID_WIDTH = 1000
GRID_HEIGHT = 1000
STEPS_PER_YEAR = 52    # 54 steps (weeks) per year
# Total number of steps (weeks) to run the simulation
SIMULATION_STEPS = STEPS_PER_YEAR * 3

# --- GIS Data Configuration ---
GALAPAGOS_GIS_FILE = "gadm41_ECU_2.json"
ISLAND_NAME_COLUMN_IN_GIS = "NAME_2"

# --- Habitat Configuration ---
HABITAT_TYPES = {
    "Sea": 0, "Coastal": 1, "Scrubland": 2, "Highland": 3
}
COASTAL_ZONE_DEPTH = 3
HIGHLAND_THRESHOLD_AREA = 50
HIGHLAND_CORE_RATIO = 0.3

# --- Species Configuration (Specified Annually) ---
# These values are now annual rates or durations in years.
# A helper function will convert them to per-step values.
SPECIES_DATA_ANNUAL = {
    "GalapagosPenguin": {  # Spheniscus mendiculus
        "annual_repro_rate": 0.50,
        "annual_mortality_base": 0.067,
        "max_age_years": 20,
        "max_energy": 110,
        "initial_energy_range": (60, 90),
        "dispersal_skill": 0.10,
        "sensing_range": 20,
        "energy_regen_factor": {
            "Coastal": 1.6,
            "Scrubland": 0.3,
            "Highland": 0.0,
            "Sea": 1.8
        },
        "base_energy_regen_land": 6,
        "habitat_preference_score": {
            "Coastal": 9,
            "Scrubland": 2,
            "Highland": 0
        },
        "move_cost_land": 5,
        "move_cost_water": 3,
        "repro_energy_cost": 45,
        "min_repro_energy": 70,

        "repro_habitat_bonus": {
            "Coastal": 0,
            "Scrubland": 0,
            "Highland": 0
        }
    },
    "FlightlessCormorant": {  # Nannopterum harrisi
        "annual_repro_rate": 0.30,
        "annual_mortality_base": 0.13,
        "max_age_years": 13,
        "max_energy": 120,
        "initial_energy_range": (70, 100),
        "dispersal_skill": 0.05,
        "sensing_range": 12,
        "energy_regen_factor": {
            "Coastal": 1.7,
            "Scrubland": 0.4,
            "Highland": 0.0,
            "Sea": 1.0
        },
        "base_energy_regen_land": 6,
        "habitat_preference_score": {
            "Coastal": 9,
            "Scrubland": 1,
            "Highland": 0
        },
        "move_cost_land": 4,
        "move_cost_water": 4,
        "repro_energy_cost": 50,
        "min_repro_energy": 80,
        "repro_habitat_bonus": {
            "Coastal": 0,
            "Scrubland": 0,
            "Highland": 0
        }
    },
    "LavaGull": {  # Leucophaeus fuliginosus
        "annual_repro_rate": 0.50,
        "annual_mortality_base": 0.067,
        "max_age_years": 20,
        "max_energy": 130,
        "initial_energy_range": (70, 110),
        "dispersal_skill": 0.40,
        "sensing_range": 18,
        "energy_regen_factor": {
            "Coastal": 1.6,
            "Scrubland": 0.7,
            "Highland": 0.3,
            "Sea": 0.3
        },
        "base_energy_regen_land": 7,
        "habitat_preference_score": {
            "Coastal": 9,
            "Scrubland": 4,
            "Highland": 1
        },
        "move_cost_land": 3,
        "move_cost_water": 4,
        "repro_energy_cost": 55,
        "min_repro_energy": 85,
        "repro_habitat_bonus": {
            "Coastal": 0,
            "Scrubland": 0,
            "Highland": 0
        }
    },

    "WavedAlbatross": {  # Phoebastria irrorata
        "annual_repro_rate": 0.50,
        "annual_mortality_base": 0.075,
        "max_age_years": 20,
        "max_energy": 160,
        "initial_energy_range": (90, 140),
        "dispersal_skill": 0.95,
        "sensing_range": 35,
        "energy_regen_factor": {
            "Coastal": 1.4,
            "Scrubland": 0.5,
            "Highland": 0.2,
            "Sea": 1.0
        },
        "base_energy_regen_land": 6,
        "habitat_preference_score": {
            "Coastal": 8,
            "Scrubland": 3,
            "Highland": 1
        },
        "move_cost_land": 6,
        "move_cost_water": 4,
        "repro_energy_cost": 70,
        "min_repro_energy": 100,
        "repro_habitat_bonus": {
            "Coastal": 0,
            "Scrubland": 0,
            "Highland": 0
        }
    },
    "GalapagosHawk": {  # Buteo galapagoensis
        "annual_repro_rate": 0.50,
        "annual_mortality_base": 0.12,
        "max_age_years": 20,
        "max_energy": 140,
        "initial_energy_range": (80, 120),
        "dispersal_skill": 0.30,
        "sensing_range": 25,
        "energy_regen_factor": {
            "Coastal": 1.0,
            "Scrubland": 1.3,
            "Highland": 1.1,
            "Sea": 0.0
        },
        "base_energy_regen_land": 8,
        "habitat_preference_score": {
            "Coastal": 6,
            "Scrubland": 8,
            "Highland": 7
        },
        "move_cost_land": 4,
        "move_cost_water": 25,
        "repro_energy_cost": 60,
        "min_repro_energy": 90,
        "repro_habitat_bonus": {
            "Coastal": 0,
            "Scrubland": 0,
            "Highland": 0
        }
    },

    "MangroveFinch": {  # Camarhynchus heliobates
        "annual_repro_rate": 0.30,
        "annual_mortality_base": 0.15,
        "max_age_years": 12,
        "max_energy": 100,
        "initial_energy_range": (55, 85),
        "dispersal_skill": 0.05,
        "sensing_range": 10,
        "energy_regen_factor": {
            "Coastal": 0.5,
            "Scrubland": 0.4,
            "Highland": 0.1,
            "Sea": 0.0
        },
        "base_energy_regen_land": 7,
        "habitat_preference_score": {
            "Coastal": 2,
            "Scrubland": 9,
            "Highland": 0
        },
        "move_cost_land": 3,
        "move_cost_water": 25,
        "repro_energy_cost": 40,
        "min_repro_energy": 60,
        "repro_habitat_bonus": {
            "Coastal": 0,
            "Scrubland": 0,
            "Highland": 0
        }
    }
}
INITIAL_POP_PER_SPECIES_PER_ISLAND = 3
# ANNUAL_MAINLAND_IMMIGRATION_RATE: Probability per species, per year an individual *might* appear.
# This will be converted to a per-step rate.
# Example: If 0.005 was intended per original more frequent step.
ANNUAL_MAINLAND_IMMIGRATION_RATE = 0.005 * STEPS_PER_YEAR
# Or, if 0.005 is a true annual rate, just use 0.005.
# Let's assume the 0.005 from before was a good *per-step* rate when steps were more frequent.
# If the *annual* desired influx is 0.005 *per island*, then this is fine.
# For clarity: If the goal is "0.005 chance per YEAR for an immigrant to appear",
# then ANNUAL_MAINLAND_IMMIGRATION_RATE = 0.005.
# The code below will divide this by STEPS_PER_YEAR.
# Let's assume the original 0.005 was a per-step rate and steps were (hypothetically) monthly.
# Then an annual rate would be ~0.005*12.
# To keep the *per-step* rate the same as in the previous iteration:
# mainland_immigration_rate_per_step = 0.005
# If we want to define it annually first, and 0.005 was good per old step:
# Let's set a plausible *annual* rate:

MAINLAND_POINT = (GRID_WIDTH + 20, GRID_HEIGHT // 2)

# --- Helper Function for Normalizing Annual Data ---


def normalize_simulation_data(annual_species_data, annual_immigration_rate, steps_per_year):
    """
    Converts annual species data and immigration rates to per-step values.
    """
    processed_species_data = {}
    for species_name, config in annual_species_data.items():
        processed_config = config.copy()  # Start with a copy of all original params

        # Convert annual rates/durations to per-step
        processed_config["repro_rate"] = config["annual_repro_rate"] / \
            steps_per_year
        processed_config["mortality_base"] = config["annual_mortality_base"] / steps_per_year
        processed_config["max_age"] = config["max_age_years"] * steps_per_year

        # Remove the original annual keys to avoid confusion if desired, or keep them
        # For clarity, let's remove them from the config passed to the agent
        del processed_config["annual_repro_rate"]
        del processed_config["annual_mortality_base"]
        del processed_config["max_age_years"]

        processed_species_data[species_name] = processed_config

    mainland_immigration_rate_per_step = annual_immigration_rate / steps_per_year

    return processed_species_data, mainland_immigration_rate_per_step

# --- Helper Functions ---


def get_habitat_name(habitat_id):
    for name, id_val in HABITAT_TYPES.items():
        if id_val == habitat_id:
            return name
    return "Unknown"

# --- Agent Definition ---


class BirdAgent(mesa.Agent):
    def __init__(self, unique_id, model, species_name, initial_pos):
        super().__init__(unique_id, model)
        self.species_name = species_name
        # self.s_data now contains per-step normalized values
        # Get processed data
        self.s_data = model.species_data_processed[species_name]
        self.age = 0  # Age in steps (weeks)
        self.energy = random.randint(*self.s_data["initial_energy_range"])
        # Ensure initial_pos is valid before accessing maps
        if 0 <= initial_pos[0] < model.grid.width and 0 <= initial_pos[1] < model.grid.height:
            self.current_island_id = self.model.island_map_ids[initial_pos[0], initial_pos[1]]
            self.current_habitat_id = self.model.habitat_map[initial_pos[0], initial_pos[1]]
        # Fallback if initial_pos is somehow out of bounds (should not happen with proper init)
        else:
            print(
                f"Warning: Agent {unique_id} ({species_name}) initialized at invalid pos {initial_pos}. Setting default island/habitat.")
            self.current_island_id = -1  # Sea
            self.current_habitat_id = HABITAT_TYPES["Sea"]

    def get_preferred_moves(self, possible_moves):
        scored_moves = []
        for move in possible_moves:
            habitat_id_at_move = self.model.habitat_map[move[0], move[1]]
            habitat_name = get_habitat_name(habitat_id_at_move)
            score = self.s_data["habitat_preference_score"].get(
                habitat_name, 0)
            if habitat_name == "Sea" and hasattr(self, 'target_island_direction') and self.target_island_direction:
                if len(self.target_island_direction) >= 2:
                    current_dx, current_dy = move[0] - \
                        self.pos[0], move[1] - self.pos[1]
                    similarity = (
                        current_dx * self.target_island_direction[0] + current_dy * self.target_island_direction[1])
                    if similarity > 0:
                        score += 2
            scored_moves.append((score, move))
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]

    def sense_islands(self):
        self.target_island_direction = None
        if self.model.habitat_map[self.pos[0], self.pos[1]] == HABITAT_TYPES["Sea"]:
            min_dist, target_island_cell = float('inf'), None
            for _ in range(8):
                angle = random.uniform(0, 2 * np.pi)
                for r in range(1, int(self.s_data["sensing_range"])):
                    check_x, check_y = int(
                        self.pos[0] + r * np.cos(angle)), int(self.pos[1] + r * np.sin(angle))
                    if 0 <= check_x < self.model.grid.width and 0 <= check_y < self.model.grid.height:
                        if self.model.island_map_types[check_x, check_y] == "land":
                            dist = np.sqrt(
                                (check_x - self.pos[0])**2 + (check_y - self.pos[1])**2)
                            if dist < min_dist:
                                min_dist, target_island_cell = dist, (
                                    check_x, check_y)
                            break
                    else:
                        break
            if target_island_cell:
                dx, dy = target_island_cell[0] - \
                    self.pos[0], target_island_cell[1] - self.pos[1]
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    self.target_island_direction = (dx/norm, dy/norm)

    def move(self):
        current_habitat_id = self.model.habitat_map[self.pos[0], self.pos[1]]
        current_habitat_name = get_habitat_name(current_habitat_id)
        possible_moves = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        valid_moves = list(possible_moves)

        if not valid_moves:
            self.energy -= self.s_data["move_cost_land"]
            return

        if current_habitat_name == "Sea":
            self.sense_islands()
        preferred_moves = self.get_preferred_moves(valid_moves)
        chosen_move, cost = None, 0

        if preferred_moves:
            if current_habitat_name != "Sea":
                land_moves = [
                    m for m in preferred_moves if self.model.habitat_map[m[0], m[1]] != HABITAT_TYPES["Sea"]]
                water_moves = [
                    m for m in preferred_moves if self.model.habitat_map[m[0], m[1]] == HABITAT_TYPES["Sea"]]
                if land_moves and (random.random() > self.s_data["dispersal_skill"] or not water_moves):
                    chosen_move = land_moves[0]
                elif water_moves and self.energy > self.s_data["move_cost_water"] * 3:
                    chosen_move = water_moves[0]
                elif land_moves:
                    chosen_move = land_moves[0]
                else:
                    chosen_move = preferred_moves[0]
            else:  # At Sea
                land_dest_moves = [
                    m for m in preferred_moves if self.model.habitat_map[m[0], m[1]] != HABITAT_TYPES["Sea"]]
                chosen_move = land_dest_moves[0] if land_dest_moves else preferred_moves[0]
        else:
            chosen_move = random.choice(valid_moves)

        if chosen_move:
            dest_hab_name = get_habitat_name(
                self.model.habitat_map[chosen_move[0], chosen_move[1]])
            cost = self.s_data["move_cost_water"] if dest_hab_name == "Sea" else self.s_data["move_cost_land"]
            if self.energy > cost:
                self.energy -= cost
                self.model.grid.move_agent(self, chosen_move)
            else:
                self.energy -= self.s_data["move_cost_land"]  # Penalty
        else:
            self.energy -= self.s_data["move_cost_land"]  # Stayed

        self.current_island_id = self.model.island_map_ids[self.pos[0], self.pos[1]]
        self.current_habitat_id = self.model.habitat_map[self.pos[0], self.pos[1]]

    def reproduce(self):
        current_habitat_name = get_habitat_name(
            self.model.habitat_map[self.pos[0], self.pos[1]])
        if current_habitat_name == "Sea":
            return

        repro_bonus = self.s_data["repro_habitat_bonus"].get(
            current_habitat_name, 1.0)
        # self.s_data["repro_rate"] is already the per-step (weekly) normalized rate
        effective_repro_rate = self.s_data["repro_rate"] * repro_bonus

        if self.energy >= self.s_data["min_repro_energy"] and random.random() < effective_repro_rate:
            self.energy -= self.s_data["repro_energy_cost"]
            possible_spawn = self.model.grid.get_neighborhood(
                self.pos, moore=True, include_center=True)
            valid_land_spawn = [
                p for p in possible_spawn if self.model.habitat_map[p[0], p[1]] != HABITAT_TYPES["Sea"]]
            spawn_point = random.choice(
                valid_land_spawn) if valid_land_spawn else self.pos
            offspring = BirdAgent(self.model.next_agent_id(
            ), self.model, self.species_name, spawn_point)
            self.model.grid.place_agent(offspring, spawn_point)
            self.model.schedule.add(offspring)

    def gain_energy(self):
        current_habitat_name = get_habitat_name(
            self.model.habitat_map[self.pos[0], self.pos[1]])
        regen_factor = self.s_data["energy_regen_factor"].get(
            current_habitat_name, 0)
        base_regen = self.s_data["base_energy_regen_land"] if current_habitat_name != "Sea" else self.s_data.get(
            "base_energy_regen_sea", 0)
        self.energy += base_regen * regen_factor
        self.energy = min(self.energy, self.s_data["max_energy"])
        self.energy = max(0, self.energy)

    def check_mortality(self):
        if self.energy <= 0 or self.age > self.s_data["max_age"]:
            return True  # max_age is in steps
        current_habitat_name = get_habitat_name(
            self.model.habitat_map[self.pos[0], self.pos[1]])
        pref_score = self.s_data["habitat_preference_score"].get(
            current_habitat_name, 0)
        mortality_mod = 2.0 if current_habitat_name == "Sea" else (
            1.5 if pref_score < 3 else 1.0)
        # self.s_data["mortality_base"] is already the per-step (weekly) normalized rate
        return random.random() < self.s_data["mortality_base"] * mortality_mod

    def step(self):
        self.age += 1
        self.move()
        self.gain_energy()
        if self.check_mortality():
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            return
        self.reproduce()

# --- Model Definition ---


class GalapagosModel(mesa.Model):
    def __init__(self, width, height, gis_file, island_name_col,
                 processed_species_data,  # Expects data processed by helper
                 initial_pop,
                 mainland_immigration_rate_per_step,  # Expects per-step rate
                 mainland_point):
        super().__init__()
        self.width, self.height = width, height
        self.gis_file, self.island_name_col = gis_file, island_name_col
        self.species_data_processed = processed_species_data  # Store the processed data
        self.initial_pop_per_species_per_island = initial_pop
        self.mainland_immigration_rate_per_step = mainland_immigration_rate_per_step
        self.mainland_point = mainland_point
        self._agent_id_counter = 0

        self.grid = mesa.space.MultiGrid(width, height, torus=False)
        self.schedule = mesa.time.RandomActivation(self)

        self.island_map_types = np.full((width, height), "sea", dtype=object)
        self.island_map_ids = np.full((width, height), -1, dtype=int)
        self.habitat_map = np.full(
            (width, height), HABITAT_TYPES["Sea"], dtype=int)
        self.islands_info = []

        self._load_and_process_gis()
        self._assign_habitats()
        self._initialize_agents()
        self._setup_datacollector()
        self.running = True
        self.datacollector.collect(self)

    def next_agent_id(self):
        self._agent_id_counter += 1
        return self._agent_id_counter

    def _load_and_process_gis(self):
        print(f"Loading GIS data from: {self.gis_file}")
        try:
            full_gdf = gpd.read_file(self.gis_file)
        except Exception as e:
            print(f"ERROR loading GIS: {e}. Continuing without islands.")
            return

        galapagos_province_name = "Galápagos"
        province_col = "NAME_1"
        try:
            gdf = full_gdf[full_gdf[province_col] == galapagos_province_name]
            if gdf.empty:
                print(
                    f"WARN: Province '{galapagos_province_name}' not found in '{province_col}'. Available: {full_gdf[province_col].unique()}. Using all features.")
                gdf = full_gdf
            else:
                print(f"Filtered for '{galapagos_province_name}'.")
        except KeyError:
            print(
                f"ERROR: Column '{province_col}' not found for province filter. Available: {full_gdf.columns.tolist()}. Using all features.")
            gdf = full_gdf

        minx, miny, maxx, maxy = gdf.total_bounds
        if gdf.empty or not all(np.isfinite([minx, miny, maxx, maxy])):
            print("ERROR: Invalid GIS bounds. Cannot process islands.")
            self.x_scale, self.y_scale, self.x_offset, self.y_offset = 1, 1, 0, 0
            return

        self.x_scale = self.width / (maxx - minx) if (maxx - minx) else 1
        self.y_scale = self.height / (maxy - miny) if (maxy - miny) else 1
        self.x_offset, self.y_offset = minx, miny
        print(
            f"GIS Bounds: ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f}). Scale: x={self.x_scale:.2f}, y={self.y_scale:.2f}")

        for idx, feat in gdf.iterrows():
            name, geom = feat.get(self.island_name_col,
                                  f"Island_{idx}"), feat.geometry
            cells = []
            for gx in range(self.width):
                for gy in range(self.height):
                    ux, uy = (gx + 0.5) / self.x_scale + \
                        self.x_offset, (gy + 0.5) / \
                        self.y_scale + self.y_offset
                    if geom.contains(Point(ux, uy)):
                        self.island_map_types[gx,
                                              gy], self.island_map_ids[gx, gy] = "land", idx
                        cells.append((gx, gy))
            if cells:
                cent_orig = geom.centroid
                gcx = max(
                    0, min(self.width - 1, int((cent_orig.x - self.x_offset) * self.x_scale)))
                gcy = max(0, min(self.height - 1,
                          int((cent_orig.y - self.y_offset) * self.y_scale)))
                self.islands_info.append({
                    "id": idx, "name": name, "geometry_orig": geom, "area_cells": len(cells),
                    "centroid_grid": (gcx, gcy), "cells_coords": cells, "coastal_cells": []
                })
                print(f"Processed: {name} (ID {idx}) with {len(cells)} cells.")
            else:
                print(f"WARN: {name} (ID {idx}) resulted in 0 grid cells.")
        if not self.islands_info:
            print("CRITICAL WARN: No islands processed from GIS.")

    def _assign_habitats(self):
        if not self.islands_info:
            return
        for r_val in range(self.height):
            for c_val in range(self.width):
                if self.island_map_types[c_val, r_val] == "land":
                    is_coastal = False
                    for dx_val in [-1, 0, 1]:
                        for dy_val in [-1, 0, 1]:
                            if dx_val == 0 and dy_val == 0:
                                continue
                            near_sea = False
                            for depth_val in range(1, COASTAL_ZONE_DEPTH + 1):
                                ncd, nrd = c_val + dx_val*depth_val, r_val + dy_val*depth_val
                                if 0 <= ncd < self.grid.width and 0 <= nrd < self.grid.height:
                                    if self.island_map_types[ncd, nrd] == "sea":
                                        near_sea = True
                                        break
                                else:
                                    near_sea = True
                                    break  # Out of bounds is like sea
                            if near_sea:
                                is_coastal = True
                                break
                        if is_coastal:
                            break

                    if is_coastal:
                        self.habitat_map[c_val,
                                         r_val] = HABITAT_TYPES["Coastal"]
                        isl_idx = self.island_map_ids[c_val, r_val]
                        for isl in self.islands_info:
                            if isl["id"] == isl_idx:
                                isl["coastal_cells"].append((c_val, r_val))
                                break
                    else:
                        isl_idx = self.island_map_ids[c_val, r_val]
                        curr_isl = next(
                            (i for i in self.islands_info if i["id"] == isl_idx), None)
                        if curr_isl and curr_isl["area_cells"] > HIGHLAND_THRESHOLD_AREA:
                            dist_c = np.sqrt(
                                (c_val - curr_isl["centroid_grid"][0])**2 + (r_val - curr_isl["centroid_grid"][1])**2)
                            rad_approx = np.sqrt(
                                curr_isl["area_cells"] / np.pi)
                            self.habitat_map[c_val, r_val] = HABITAT_TYPES["Highland"] if dist_c < rad_approx * \
                                HIGHLAND_CORE_RATIO else HABITAT_TYPES["Scrubland"]
                        else:
                            self.habitat_map[c_val,
                                             r_val] = HABITAT_TYPES["Scrubland"]
                else:
                    self.habitat_map[c_val, r_val] = HABITAT_TYPES["Sea"]

    def _initialize_agents(self):
        if not self.islands_info:
            return
        for island in self.islands_info:
            if not island["cells_coords"]:
                continue
            spawn_cells = [c for c in island["cells_coords"] if self.habitat_map[c[0], c[1]] in [
                HABITAT_TYPES["Coastal"], HABITAT_TYPES["Scrubland"]]]
            if not spawn_cells:
                spawn_cells = island["cells_coords"]
            if not spawn_cells:
                continue

            for species_name in self.species_data_processed.keys():  # Use processed data keys
                for _ in range(self.initial_pop_per_species_per_island):
                    pos = random.choice(spawn_cells)
                    agent = BirdAgent(self.next_agent_id(),
                                      self, species_name, pos)
                    self.grid.place_agent(agent, pos)
                    self.schedule.add(agent)

    def _setup_datacollector(self):
        model_reporters = {
            "TotalAgents": lambda m: m.schedule.get_agent_count()}
        for s_name in self.species_data_processed.keys():  # Use processed data keys
            model_reporters[f"{s_name}_total"] = lambda m, s=s_name: sum(
                1 for a in m.schedule.agents if isinstance(a, BirdAgent) and a.species_name == s)
        if self.islands_info:
            for i_info in self.islands_info:
                i_id, i_name = i_info["id"], i_info["name"].replace(" ", "_")
                model_reporters[f"Richness_{i_name}"] = lambda m, id_val=i_id: len(set(a.species_name for a in m.schedule.agents if isinstance(
                    a, BirdAgent) and hasattr(a, 'current_island_id') and a.current_island_id == id_val))
                for s_name in self.species_data_processed.keys():
                    model_reporters[f"Pop_{i_name}_{s_name}"] = lambda m, id_val=i_id, sn=s_name: sum(1 for a in m.schedule.agents if isinstance(
                        a, BirdAgent) and hasattr(a, 'current_island_id') and a.current_island_id == id_val and a.species_name == sn)
        else:
            print("WARN: No islands, limited data collection.")
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters)

    def apply_mainland_immigration(self):
        if not self.islands_info:
            return
        for island in self.islands_info:
            target_cells = island.get(
                "coastal_cells", []) or island.get("cells_coords", [])
            if not target_cells:
                continue
            for species_name in self.species_data_processed.keys():
                # self.mainland_immigration_rate_per_step is already per-step
                effective_rate = self.mainland_immigration_rate_per_step * \
                    (len(target_cells) / 50.0)  # Size factor
                if random.random() < effective_rate:
                    pos = random.choice(target_cells)
                    immigrant = BirdAgent(
                        self.next_agent_id(), self, species_name, pos)
                    self.grid.place_agent(immigrant, pos)
                    self.schedule.add(immigrant)

    def step(self):
        if self.schedule.get_agent_count() > 0:
            self.apply_mainland_immigration()
            self.schedule.step()
        else:
            print(f"Step {self.schedule.steps}: No agents.")
        self.datacollector.collect(self)
        if self.schedule.steps % 50 == 0:
            print(
                f"--- Step {self.schedule.steps} (Week) --- Agents: {self.schedule.get_agent_count()} ---")

# --- Validation and Plotting Functions (mostly unchanged, check labels) ---


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


# --- Simulation Run & Visualization ---
if __name__ == "__main__":
    print("Starting Galapagos ABM Simulation...")
    print(
        f"Simulation configured for {STEPS_PER_YEAR} steps (weeks) per year.")
    print(
        f"Total simulation duration: {SIMULATION_STEPS} weeks (approx. {SIMULATION_STEPS/STEPS_PER_YEAR:.2f} years).")

    # Normalize annual data to per-step data for the simulation
    processed_species_data_for_model, mainland_immigration_rate_for_model = normalize_simulation_data(
        SPECIES_DATA_ANNUAL,
        ANNUAL_MAINLAND_IMMIGRATION_RATE,
        STEPS_PER_YEAR
    )

    print(
        f"Per-step mainland immigration rate: {mainland_immigration_rate_for_model:.6f}")
    # Example: print processed data for one species to verify
    # print("Processed GroundFinch data for model:", processed_species_data_for_model["GroundFinch"])

    model_instance = GalapagosModel(
        GRID_WIDTH, GRID_HEIGHT,
        GALAPAGOS_GIS_FILE, ISLAND_NAME_COLUMN_IN_GIS,
        processed_species_data_for_model,  # Pass the processed data
        INITIAL_POP_PER_SPECIES_PER_ISLAND,
        mainland_immigration_rate_for_model,  # Pass the processed rate
        MAINLAND_POINT
    )

    if not model_instance.islands_info:
        print("CRITICAL: No islands loaded. Check GIS path/content.")
    else:
        print(
            f"Model initialized with {len(model_instance.islands_info)} islands.")

    for i in range(SIMULATION_STEPS):
        model_instance.step()
        if model_instance.schedule.get_agent_count() == 0 and i > 20:
            print(f"Stopping early at week {i+1}: all agents perished.")
            break

    print("\nSimulation finished.")
    model_run_data = model_instance.datacollector.get_model_vars_dataframe()
    print("\n--- Simulation Results (Last 5 steps) ---")
    print(model_run_data.tail())

    plt.figure(figsize=(12, 6))
    # Use processed_species_data_for_model to iterate for consistency if SPECIES_DATA_ANNUAL was changed
    for s_name in processed_species_data_for_model.keys():
        col = f"{s_name}_total"
        if col in model_run_data.columns:
            plt.plot(model_run_data.index,
                     model_run_data[col], label=f"{s_name} Total Pop")
    plt.xlabel("Step (Week)")
    plt.ylabel("Total Population")
    plt.title("Overall Species Populations Over Time")
    plt.legend()
    plt.grid(True, ls=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    if model_instance.islands_info:
        plt.figure(figsize=(12, 6))
        for isl in model_instance.islands_info:
            col = f"Richness_{isl['name'].replace(' ', '_')}"
            if col in model_run_data.columns:
                plt.plot(model_run_data.index,
                         model_run_data[col], label=f"{isl['name']} Richness")
        plt.xlabel("Step (Week)")
        plt.ylabel("Species Richness")
        plt.title("Island Species Richness Over Time")
        plt.legend(fontsize='small', ncol=2)
        plt.grid(True, ls=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    if model_instance.islands_info and not model_run_data.empty:
        plot_species_area_relationship(
            model_run_data, model_instance.islands_info)
        plot_species_isolation_relationship(
            model_run_data, model_instance.islands_info, MAINLAND_POINT)
    else:
        print("Skipping SAR/SIR plots: no island data or model run data.")

    plot_final_distribution(model_instance)
