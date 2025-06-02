import mesa
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

# --- Configuration ---
GRID_WIDTH = 100  # Increased grid size for more detailed islands
GRID_HEIGHT = 100
SIMULATION_STEPS = 300 # Number of steps to run the simulation

# --- GIS Data Configuration ---
# IMPORTANT: Replace with the actual path to your Galapagos Islands shapefile or GeoJSON
GALAPAGOS_GIS_FILE = "gadm41_ECU_2.json" # or .shp
# Example: If you download from GADM for Ecuador, then select Galapagos.
# Or use Natural Earth data. Ensure it has polygons for individual islands.
# The GIS file should have a column that can be used for island names, e.g., 'NAME', 'island_name'
ISLAND_NAME_COLUMN_IN_GIS = "NAME_2" # Adjust based on your GIS file's attributes

# --- Habitat Configuration ---
HABITAT_TYPES = {
    "Sea": 0, # Numerical ID for sea
    "Coastal": 1,
    "Scrubland": 2,
    "Highland": 3
}
COASTAL_ZONE_DEPTH = 3 # How many cells inland from the sea is 'Coastal'
HIGHLAND_THRESHOLD_AREA = 100 # Minimum island area (in cells) to have a 'Highland' zone (conceptual)
HIGHLAND_CORE_RATIO = 0.3 # Ratio of island radius for highland core (conceptual)


# --- Species Configuration (Dummy Data) ---
SPECIES_DATA = {
    "GroundFinch": {
        "repro_rate": 0.12, "mortality_base": 0.025, "max_age": 10, "max_energy": 100,
        "initial_energy_range": (50, 80),
        "dispersal_skill": 0.2, "sensing_range": 15, # Sensing range in grid cells
        "energy_regen_factor": {"Coastal": 1.1, "Scrubland": 1.5, "Highland": 0.8, "Sea": 0}, # Multiplier for base regen
        "base_energy_regen_land": 8,
        "habitat_preference_score": {"Coastal": 5, "Scrubland": 8, "Highland": 2}, # Higher is better
        "move_cost_land": 2, "move_cost_water": 20,
        "repro_energy_cost": 40, "min_repro_energy": 60,
        "repro_habitat_bonus": {"Coastal": 1.0, "Scrubland": 1.2, "Highland": 0.7} # Multiplier for repro_rate
    },
    "MarineIguana": { # Example of a more specialized species
        "repro_rate": 0.08, "mortality_base": 0.015, "max_age": 12, "max_energy": 120,
        "initial_energy_range": (60, 100),
        "dispersal_skill": 0.7, "sensing_range": 10,
        "energy_regen_factor": {"Coastal": 1.8, "Scrubland": 0.5, "Highland": 0.1, "Sea": 0.2}, # Can get some from sea
        "base_energy_regen_land": 6, # Lower on land, higher in coastal
        "habitat_preference_score": {"Coastal": 10, "Scrubland": 1, "Highland": 0},
        "move_cost_land": 3, "move_cost_water": 8,
        "repro_energy_cost": 50, "min_repro_energy": 70,
        "repro_habitat_bonus": {"Coastal": 1.5, "Scrubland": 0.5, "Highland": 0.1}
    },
    "BlueFootedBooby": {
        "repro_rate": 0.06, "mortality_base": 0.01, "max_age": 15, "max_energy": 150,
        "initial_energy_range": (80, 130),
        "dispersal_skill": 0.9, "sensing_range": 30,
        "energy_regen_factor": {"Coastal": 1.6, "Scrubland": 0.8, "Highland": 0.5, "Sea": 0.1},
        "base_energy_regen_land": 7,
        "habitat_preference_score": {"Coastal": 9, "Scrubland": 3, "Highland": 1},
        "move_cost_land": 4, "move_cost_water": 5,
        "repro_energy_cost": 60, "min_repro_energy": 80,
        "repro_habitat_bonus": {"Coastal": 1.3, "Scrubland": 0.8, "Highland": 0.4}
    }
}
INITIAL_POP_PER_SPECIES_PER_ISLAND = 3
MAINLAND_IMMIGRATION_RATE = 0.005 # Per species, per step, an individual *might* appear on a random coastal cell
MAINLAND_POINT = (GRID_WIDTH + 20, GRID_HEIGHT // 2) # Conceptual point for isolation calculation

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
        self.s_data = SPECIES_DATA[species_name]
        self.age = 0
        self.energy = random.randint(*self.s_data["initial_energy_range"])
        self.current_island_id = self.model.island_map_ids[initial_pos[0], initial_pos[1]]
        self.current_habitat_id = self.model.habitat_map[initial_pos[0], initial_pos[1]]

    def get_preferred_moves(self, possible_moves):
        """Return moves sorted by habitat preference."""
        scored_moves = []
        for move in possible_moves: # 'move' is already guaranteed to be within bounds here
            habitat_id_at_move = self.model.habitat_map[move[0], move[1]]
            habitat_name = get_habitat_name(habitat_id_at_move)
            score = self.s_data["habitat_preference_score"].get(habitat_name, 0)
            
            # Add a small bonus for moving towards a sensed island if in sea
            if habitat_name == "Sea" and hasattr(self, 'target_island_direction'):
                # Ensure target_island_direction is not None before trying to access its elements
                if self.target_island_direction: 
                    current_dx = move[0] - self.pos[0]
                    current_dy = move[1] - self.pos[1]
                    # Check if target_island_direction has at least two elements
                    if len(self.target_island_direction) >= 2:
                        similarity = (current_dx * self.target_island_direction[0] + 
                                      current_dy * self.target_island_direction[1])
                        if similarity > 0: 
                            score += 2 
            scored_moves.append((score, move))
        
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]

    def sense_islands(self):
        """Simplified island sensing mechanism."""
        self.target_island_direction = None
        if self.model.habitat_map[self.pos[0], self.pos[1]] == HABITAT_TYPES["Sea"]:
            min_dist = float('inf')
            target_island_cell = None
            
            # Check a few random directions or a sparse grid for distant land
            for _ in range(8): # Check 8 directions
                angle = random.uniform(0, 2 * np.pi)
                for r in range(1, int(self.s_data["sensing_range"])):
                    check_x = int(self.pos[0] + r * np.cos(angle))
                    check_y = int(self.pos[1] + r * np.sin(angle))

                    # Manually check if the calculated (check_x, check_y) are within grid boundaries
                    if (0 <= check_x < self.model.grid.width and 0 <= check_y < self.model.grid.height):
                        if self.model.island_map_types[check_x, check_y] == "land":
                            dist = np.sqrt((check_x - self.pos[0])**2 + (check_y - self.pos[1])**2)
                            if dist < min_dist:
                                min_dist = dist
                                target_island_cell = (check_x, check_y)
                            break # Found land in this direction
                    else: # This 'else' now correctly means the coordinates were out of bounds
                        break # Stop searching in this direction if out of bounds

            if target_island_cell:
                dx = target_island_cell[0] - self.pos[0]
                dy = target_island_cell[1] - self.pos[1]
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    self.target_island_direction = (dx/norm, dy/norm)


    def move(self):
        current_habitat_id = self.model.habitat_map[self.pos[0], self.pos[1]]
        current_habitat_name = get_habitat_name(current_habitat_id)

        possible_moves = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        valid_moves = list(possible_moves)

        if not valid_moves:
            self.energy -= self.s_data["move_cost_land"] # Cost for staying
            return

        # Sensing for sophisticated dispersal
        if current_habitat_name == "Sea":
            self.sense_islands()

        preferred_moves = self.get_preferred_moves(valid_moves)
        
        chosen_move = None
        cost = 0

        # Try to move based on preference
        if preferred_moves:
            # If on land, prefer land. If at sea, try to reach land or follow sensed direction.
            if current_habitat_name != "Sea":
                # Prefer land moves unless dispersal skill is high and water is an option
                land_preferred_moves = [m for m in preferred_moves if self.model.habitat_map[m[0],m[1]] != HABITAT_TYPES["Sea"]]
                water_preferred_moves = [m for m in preferred_moves if self.model.habitat_map[m[0],m[1]] == HABITAT_TYPES["Sea"]]

                if land_preferred_moves and (random.random() > self.s_data["dispersal_skill"] or not water_preferred_moves):
                    chosen_move = land_preferred_moves[0]
                elif water_preferred_moves and self.energy > self.s_data["move_cost_water"] * 3: # Need more energy to attempt sea
                    chosen_move = water_preferred_moves[0]
                elif land_preferred_moves: # Fallback
                    chosen_move = land_preferred_moves[0]
                else: # No good land option, try water if available
                    chosen_move = preferred_moves[0]
            
            else: # Currently at Sea
                # Prioritize land if available in preferred_moves
                land_destination_moves = [m for m in preferred_moves if self.model.habitat_map[m[0],m[1]] != HABITAT_TYPES["Sea"]]
                if land_destination_moves:
                    chosen_move = land_destination_moves[0]
                else: # Stay at sea, follow preference (which might include sensed direction bonus)
                    chosen_move = preferred_moves[0]
        else: # No preferred moves (e.g. stuck), pick a random valid one
            chosen_move = random.choice(valid_moves)


        if chosen_move:
            destination_habitat_id = self.model.habitat_map[chosen_move[0], chosen_move[1]]
            destination_habitat_name = get_habitat_name(destination_habitat_id)

            if destination_habitat_name == "Sea":
                cost = self.s_data["move_cost_water"]
            else:
                cost = self.s_data["move_cost_land"]
            
            if self.energy > cost:
                self.energy -= cost
                self.model.grid.move_agent(self, chosen_move)
            else:
                self.energy -= self.s_data["move_cost_land"] # Penalty for failed attempt
        else: # Stayed put
             self.energy -= self.s_data["move_cost_land"]


        self.current_island_id = self.model.island_map_ids[self.pos[0], self.pos[1]]
        self.current_habitat_id = self.model.habitat_map[self.pos[0], self.pos[1]]


    def reproduce(self):
        current_habitat_name = get_habitat_name(self.model.habitat_map[self.pos[0], self.pos[1]])
        
        if current_habitat_name == "Sea": # Cannot reproduce in the sea
            return

        repro_bonus = self.s_data["repro_habitat_bonus"].get(current_habitat_name, 1.0)
        effective_repro_rate = self.s_data["repro_rate"] * repro_bonus

        if self.energy >= self.s_data["min_repro_energy"] and \
           random.random() < effective_repro_rate:
            
            self.energy -= self.s_data["repro_energy_cost"]
            
            possible_spawn_points = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=True)
            valid_land_spawn_points = [
                p for p in possible_spawn_points 
                # 'p' from get_neighborhood (non-torus grid) is already guaranteed to be within bounds.
                # We only need to check if it's not sea.
                if self.model.habitat_map[p[0],p[1]] != HABITAT_TYPES["Sea"]
            ]
            
            spawn_point = random.choice(valid_land_spawn_points) if valid_land_spawn_points else self.pos

            new_id = self.model.next_agent_id()
            offspring = BirdAgent(new_id, self.model, self.species_name, spawn_point)
            self.model.grid.place_agent(offspring, spawn_point)
            self.model.schedule.add(offspring)

    def gain_energy(self):
        current_habitat_name = get_habitat_name(self.model.habitat_map[self.pos[0], self.pos[1]])
        regen_factor = self.s_data["energy_regen_factor"].get(current_habitat_name, 0)
        
        if current_habitat_name != "Sea":
            self.energy += self.s_data["base_energy_regen_land"] * regen_factor
        else: # Specific energy gain/loss for sea if defined (e.g. MarineIguana)
            self.energy += self.s_data.get("base_energy_regen_sea", 0) * regen_factor


        self.energy = min(self.energy, self.s_data["max_energy"])
        self.energy = max(0, self.energy) # Cannot go below 0

    def check_mortality(self):
        if self.energy <= 0:
            return True
        if self.age > self.s_data["max_age"]:
            return True
        # Mortality slightly increased in non-preferred or sea habitats
        current_habitat_name = get_habitat_name(self.model.habitat_map[self.pos[0], self.pos[1]])
        preference_score = self.s_data["habitat_preference_score"].get(current_habitat_name, 0)
        
        mortality_modifier = 1.0
        if current_habitat_name == "Sea":
            mortality_modifier = 2.0 # Higher risk in sea unless specialized
        elif preference_score < 3: # Low preference
            mortality_modifier = 1.5

        if random.random() < self.s_data["mortality_base"] * mortality_modifier:
            return True
        return False

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
    def __init__(self, width, height, gis_file, island_name_col, species_data, 
                 initial_pop, mainland_immigration_rate, mainland_point):
        super().__init__()
        self.width = width
        self.height = height
        self.gis_file = gis_file
        self.island_name_col = island_name_col
        self.species_data = species_data
        self.initial_pop_per_species_per_island = initial_pop
        self.mainland_immigration_rate = mainland_immigration_rate
        self.mainland_point = mainland_point # For isolation calculation
        self._agent_id_counter = 0

        self.grid = mesa.space.MultiGrid(width, height, torus=False) # Torus false for distinct islands
        self.schedule = mesa.time.RandomActivation(self)
        
        # Initialize maps
        self.island_map_types = np.full((width, height), "sea", dtype=object) # land or sea
        self.island_map_ids = np.full((width, height), -1, dtype=int) # Island index, -1 for sea
        self.habitat_map = np.full((width, height), HABITAT_TYPES["Sea"], dtype=int)
        
        self.islands_info = [] # List of dicts: {id, name, geometry, area_km2, area_cells, centroid, cells_coords, coastal_cells}

        self._load_and_process_gis()
        self._assign_habitats()
        self._initialize_agents()
        
        self._setup_datacollector()
        
        self.running = True 
        self.datacollector.collect(self)

    def next_agent_id(self):
        self._agent_id_counter += 1
        return self._agent_id_counter

    # INSIDE THE GalapagosModel CLASS, MODIFY _load_and_process_gis:

    def _load_and_process_gis(self):
        print(f"Loading GIS data from: {self.gis_file}")
        try:
            full_gdf = gpd.read_file(self.gis_file) # Load the full Ecuador GeoDataFrame
        except Exception as e:
            print(f"ERROR: Could not load GIS file. Ensure the path is correct and it's a valid GeoJSON/Shapefile.")
            print(f"Details: {e}")
            print("Continuing with no islands. Simulation will likely be empty.")
            return

        # Filter for Galapagos Province first (Adjust 'NAME_1' and 'Galápagos' as needed based on your file)
        # You might need to inspect your GADM file to find the exact column name and value for Galapagos province
        # Common names for province column: NAME_1, ADM1_ES, etc.
        # Common names for Galapagos: Galápagos, Galapagos
        galapagos_province_name_in_file = "Galápagos" # Example, VERIFY THIS in your GADM file
        province_column_name_in_file = "NAME_1"      # Example, VERIFY THIS

        try:
            gdf = full_gdf[full_gdf[province_column_name_in_file] == galapagos_province_name_in_file]
            if gdf.empty:
                print(f"WARNING: Could not find province '{galapagos_province_name_in_file}' using column '{province_column_name_in_file}'.")
                print(f"Available unique values in '{province_column_name_in_file}': {full_gdf[province_column_name_in_file].unique()}")
                print("Attempting to process all features, which might be incorrect for Galapagos simulation.")
                gdf = full_gdf # Fallback to using all data if filter fails, but this is not ideal
            else:
                print(f"Successfully filtered for '{galapagos_province_name_in_file}' province.")
        except KeyError:
            print(f"ERROR: Column '{province_column_name_in_file}' not found in GIS file for province filtering.")
            print(f"Available columns: {full_gdf.columns.tolist()}")
            print("Attempting to process all features, which might be incorrect for Galapagos simulation.")
            gdf = full_gdf # Fallback


        # Reproject if needed (optional, but good for consistent area/distance)
        # gdf = gdf.to_crs("EPSG:32717") # UTM zone for Galapagos South, for accurate area if needed
                                    # Be careful with this if your display expects lat/lon

        minx, miny, maxx, maxy = gdf.total_bounds

        # Check if bounds are valid (e.g. if filtering failed and gdf is empty)
        if gdf.empty or not (np.isfinite(minx) and np.isfinite(miny) and np.isfinite(maxx) and np.isfinite(maxy)):
            print("ERROR: Invalid bounds after attempting to filter for Galapagos. Cannot proceed with GIS processing.")
            # Set default scales to prevent division by zero, though this won't create islands
            self.x_scale, self.y_scale = 1.0, 1.0 
            self.x_offset, self.y_offset = 0.0, 0.0
            return

        # Scale factor to map GIS coords to grid coords
        self.x_scale = self.width / (maxx - minx) if (maxx - minx) != 0 else 1
        self.y_scale = self.height / (maxy - miny) if (maxy - miny) != 0 else 1
        self.x_offset = minx
        self.y_offset = miny

        print(f"Filtered GIS Bounds (Galapagos): ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})")
        print(f"Grid Scale: x_scale={self.x_scale:.2f}, y_scale={self.y_scale:.2f}")

        for idx, island_feature in gdf.iterrows(): # gdf is now (ideally) the filtered Galapagos data
            # The ISLAND_NAME_COLUMN_IN_GIS should refer to the column for individual island names
            # within the Galapagos province (e.g., 'NAME_2' for cantons like Isabela, Santa Cruz)
            island_name = island_feature.get(self.island_name_col, f"Island_{idx}")
            island_geom = island_feature.geometry

            island_cells_coords = []
            # Rasterize: Check center of each grid cell
            # Iterate over a bounding box of the specific island_geom for efficiency (optional optimization)
            # For simplicity here, still iterating over the whole grid
            for gx in range(self.width):
                for gy in range(self.height):
                    # Convert grid cell center (gx+0.5, gy+0.5) back to original GIS-like coordinates
                    # This transformation maps grid cell centers to points in the original coordinate system of the filtered gdf
                    unscaled_x = (gx + 0.5) / self.x_scale + self.x_offset
                    unscaled_y = (gy + 0.5) / self.y_scale + self.y_offset
                    cell_point = Point(unscaled_x, unscaled_y)

                    if island_geom.contains(cell_point):
                        self.island_map_types[gx, gy] = "land"
                        # Use a unique ID for each island within the filtered set, 
                        # original 'idx' from gdf.iterrows() is fine if gdf is filtered
                        self.island_map_ids[gx, gy] = idx 
                        island_cells_coords.append((gx, gy))

            if island_cells_coords:
                # Centroid calculation needs to use the original geometry for accuracy,
                # then transform to grid coordinates
                centroid_orig_geom = island_geom.centroid # This is in the original CRS of island_geom

                # Transform this original centroid to grid coordinates
                grid_centroid_x = int((centroid_orig_geom.x - self.x_offset) * self.x_scale)
                grid_centroid_y = int((centroid_orig_geom.y - self.y_offset) * self.y_scale)
                # Clamp to grid bounds to be safe
                grid_centroid_x = max(0, min(self.width - 1, grid_centroid_x))
                grid_centroid_y = max(0, min(self.height - 1, grid_centroid_y))


                self.islands_info.append({
                    "id": idx, # This ID comes from the filtered GeoDataFrame's index
                    "name": island_name, 
                    "geometry_orig": island_geom,
                    "area_cells": len(island_cells_coords),
                    "centroid_grid": (grid_centroid_x, grid_centroid_y),
                    "cells_coords": island_cells_coords,
                    "coastal_cells": []
                })
                print(f"Processed: {island_name} (Original GDF Index {idx}) with {len(island_cells_coords)} cells.")
            else:
                print(f"Warning: Island {island_name} (Original GDF Index {idx}) resulted in 0 cells on the grid after filtering and scaling.")

        if not self.islands_info:
            print("CRITICAL WARNING: No islands were successfully processed from the GIS file after filtering for Galapagos.")

    def _assign_habitats(self):
        if not self.islands_info: return # No islands to assign habitats to

        for r in range(self.height):
            for c in range(self.width):
                if self.island_map_types[c,r] == "land":
                    is_coastal = False
                    # Check neighbors for sea
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            nc, nr = c + dx, r + dy
                            # Check within coastal depth for sea
                            is_near_sea = False
                            for depth in range(1, COASTAL_ZONE_DEPTH + 1):
                                ncd, nrd = c + dx*depth, r + dy*depth
                                if (0 <= ncd < self.grid.width and 0 <= nrd < self.grid.height): # <--- THIS IS THE FIX
                                    if self.island_map_types[ncd, nrd] == "sea":
                                        is_near_sea = True
                                        break
                                else: # Out of bounds is like sea for this check (i.e., if not within bounds)
                                    is_near_sea = True
                                    break
                            if is_near_sea:
                                is_coastal = True
                                break
                        if is_coastal: break
                    
                    if is_coastal:
                        self.habitat_map[c,r] = HABITAT_TYPES["Coastal"]
                        # Add to island's coastal cell list
                        island_idx = self.island_map_ids[c,r]
                        for isl_info in self.islands_info:
                            if isl_info["id"] == island_idx:
                                isl_info["coastal_cells"].append((c,r))
                                break
                    else:
                        # Simplistic Highland/Scrubland based on island size and centrality
                        # This is highly conceptual without elevation data
                        island_idx = self.island_map_ids[c,r]
                        current_island_info = next((isl for isl in self.islands_info if isl["id"] == island_idx), None)
                        
                        if current_island_info and current_island_info["area_cells"] > HIGHLAND_THRESHOLD_AREA:
                            dist_to_centroid = np.sqrt(
                                (c - current_island_info["centroid_grid"][0])**2 + 
                                (r - current_island_info["centroid_grid"][1])**2
                            )
                            # Approximate island "radius"
                            island_radius_approx = np.sqrt(current_island_info["area_cells"] / np.pi) 
                            if dist_to_centroid < island_radius_approx * HIGHLAND_CORE_RATIO:
                                self.habitat_map[c,r] = HABITAT_TYPES["Highland"]
                            else:
                                self.habitat_map[c,r] = HABITAT_TYPES["Scrubland"]
                        else:
                             self.habitat_map[c,r] = HABITAT_TYPES["Scrubland"] # Default for smaller islands or non-coastal
                else: # Sea
                    self.habitat_map[c,r] = HABITAT_TYPES["Sea"]


    def _initialize_agents(self):
        if not self.islands_info: return

        for island_info_dict in self.islands_info:
            if not island_info_dict["cells_coords"]: continue

            # Prefer spawning in coastal or scrubland initially
            spawnable_cells = [
                cell for cell in island_info_dict["cells_coords"]
                if self.habitat_map[cell[0], cell[1]] in [HABITAT_TYPES["Coastal"], HABITAT_TYPES["Scrubland"]]
            ]
            if not spawnable_cells: # Fallback to any land cell if no preferred habitats
                spawnable_cells = island_info_dict["cells_coords"]
            
            if not spawnable_cells: continue # Should not happen if cells_coords exists

            for species_name_key in self.species_data.keys():
                for _ in range(self.initial_pop_per_species_per_island):
                    pos = random.choice(spawnable_cells)
                    agent = BirdAgent(self.next_agent_id(), self, species_name_key, pos)
                    self.grid.place_agent(agent, pos)
                    self.schedule.add(agent)

    def _setup_datacollector(self):
        model_reporters = {"TotalAgents": lambda m: m.schedule.get_agent_count()}
        for s_name in self.species_data.keys():
            model_reporters[f"{s_name}_total"] = \
                lambda m, s=s_name: sum(1 for a in m.schedule.agents if isinstance(a, BirdAgent) and a.species_name == s)
        
        if self.islands_info: # Only add island reporters if islands were processed
            for i_info in self.islands_info:
                island_id_val = i_info["id"]
                island_name_val = i_info["name"].replace(" ", "_") # Make names filename-friendly

                model_reporters[f"Richness_{island_name_val}"] = \
                    lambda m, iid=island_id_val: len(set(a.species_name for a in m.schedule.agents if isinstance(a, BirdAgent) and hasattr(a, 'current_island_id') and a.current_island_id == iid))
                
                for s_name_key in self.species_data.keys():
                    model_reporters[f"Pop_{island_name_val}_{s_name_key}"] = \
                        lambda m, iid=island_id_val, sn=s_name_key: sum(1 for a in m.schedule.agents if isinstance(a, BirdAgent) and hasattr(a, 'current_island_id') and a.current_island_id == iid and a.species_name == sn)
        else:
            print("Warning: No islands processed. Data collector will have limited island-specific metrics.")

        self.datacollector = mesa.DataCollector(model_reporters=model_reporters)


    def apply_mainland_immigration(self):
        if not self.islands_info: return

        for island_info_item in self.islands_info:
            # Immigrate only to coastal cells if available
            target_cells = island_info_item.get("coastal_cells", [])
            if not target_cells: # Fallback to any cell on the island
                target_cells = island_info_item.get("cells_coords", [])
            if not target_cells: continue

            for species_name_val in self.species_data.keys():
                # Immigration chance slightly modified by island size (more coastal cells = bigger target)
                effective_immigration_rate = self.mainland_immigration_rate * (len(target_cells) / 50.0) # Normalize factor
                if random.random() < effective_immigration_rate :
                    pos = random.choice(target_cells)
                    immigrant = BirdAgent(self.next_agent_id(), self, species_name_val, pos)
                    self.grid.place_agent(immigrant, pos)
                    self.schedule.add(immigrant)
                    # print(f"Immigrant {species_name_val} to {island_info_item['name']} at {pos}")


    def step(self):
        if self.schedule.get_agent_count() > 0 : # Only run if agents exist
             self.apply_mainland_immigration()
             self.schedule.step()
        else: # No agents, maybe stop simulation or just note it
            print(f"Step {self.schedule.steps}: No agents remaining.")
            # self.running = False # Optional: stop if all agents die
        
        self.datacollector.collect(self)
        if self.schedule.steps % 50 == 0:
            print(f"--- Step {self.schedule.steps} --- Agents: {self.schedule.get_agent_count()} ---")


# --- Validation and Plotting Functions ---
def plot_species_area_relationship(model_data_df, islands_info_list):
    if not islands_info_list:
        print("SAR Plot: No island info available.")
        return
        
    final_richness = []
    island_areas_cells = []
    island_names_sar = []

    for isl_info in islands_info_list:
        richness_col = f"Richness_{isl_info['name'].replace(' ', '_')}"
        if richness_col in model_data_df.columns:
            # Use average richness over last N steps or final richness
            # final_richness.append(model_data_df[richness_col].iloc[-1]) 
            last_n_steps = min(50, len(model_data_df)) # Avg over last 50 or fewer if less steps
            avg_richness = model_data_df[richness_col].tail(last_n_steps).mean()
            final_richness.append(avg_richness)
            island_areas_cells.append(isl_info["area_cells"])
            island_names_sar.append(isl_info["name"])
        else:
            print(f"SAR Plot: Richness column {richness_col} not found in model data.")


    if not final_richness or not island_areas_cells:
        print("SAR Plot: Not enough data to plot SAR.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(island_areas_cells, final_richness, color='blue')
    for i, name in enumerate(island_names_sar):
        plt.annotate(name, (island_areas_cells[i], final_richness[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)
    
    # Log-log scale is common for SAR
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Island Area (Number of Cells) - Log Scale")
    plt.ylabel("Species Richness (Avg. Last 50 Steps) - Log Scale")
    plt.title("Species-Area Relationship (SAR)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_species_isolation_relationship(model_data_df, islands_info_list, mainland_ref_point):
    if not islands_info_list:
        print("SIR Plot: No island info available.")
        return

    final_richness_sir = []
    isolation_distances = []
    island_names_sir = []

    for isl_info in islands_info_list:
        richness_col = f"Richness_{isl_info['name'].replace(' ', '_')}"
        if richness_col in model_data_df.columns:
            last_n_steps = min(50, len(model_data_df))
            avg_richness = model_data_df[richness_col].tail(last_n_steps).mean()
            final_richness_sir.append(avg_richness)
            
            # Calculate distance from island centroid to mainland_ref_point
            dist = np.sqrt(
                (isl_info["centroid_grid"][0] - mainland_ref_point[0])**2 +
                (isl_info["centroid_grid"][1] - mainland_ref_point[1])**2
            )
            isolation_distances.append(dist)
            island_names_sir.append(isl_info["name"])
        else:
            print(f"SIR Plot: Richness column {richness_col} not found in model data.")
    
    if not final_richness_sir or not isolation_distances:
        print("SIR Plot: Not enough data to plot SIR.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(isolation_distances, final_richness_sir, color='green')
    for i, name in enumerate(island_names_sir):
        plt.annotate(name, (isolation_distances[i], final_richness_sir[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    plt.xlabel("Isolation (Distance to Conceptual Mainland Point)")
    plt.ylabel("Species Richness (Avg. Last 50 Steps)")
    plt.title("Species-Isolation Relationship (SIR)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_final_distribution(model):
    plt.figure(figsize=(12, 12 * (model.height / model.width))) # Adjust aspect ratio
    
    # Background map: Habitats
    habitat_color_map_def = {
        HABITAT_TYPES["Sea"]: [0.7, 0.85, 1.0],      # Light blue
        HABITAT_TYPES["Coastal"]: [0.9, 0.85, 0.7],  # Sandy yellow
        HABITAT_TYPES["Scrubland"]: [0.6, 0.8, 0.6], # Light green
        HABITAT_TYPES["Highland"]: [0.4, 0.6, 0.4]   # Darker green
    }
    
    # Create an RGB image for habitats
    habitat_display_rgb = np.zeros((model.height, model.width, 3))
    for r_val in range(model.height):
        for c_val in range(model.width):
            habitat_id = model.habitat_map[c_val, r_val]
            habitat_display_rgb[r_val, c_val, :] = habitat_color_map_def.get(habitat_id, [0,0,0]) # Black for unknown

    plt.imshow(habitat_display_rgb, origin="lower", extent=[0, model.width, 0, model.height])

    # Overlay agents
    species_color_palette_viz = {"GroundFinch": "red", "MarineIguana": "purple", "BlueFootedBooby": "cyan"}
    agent_x, agent_y, agent_c = [], [], []

    for agent_item_viz in model.schedule.agents:
        if isinstance(agent_item_viz, BirdAgent) and agent_item_viz.pos is not None:
            agent_x.append(agent_item_viz.pos[0] + 0.5)
            agent_y.append(agent_item_viz.pos[1] + 0.5)
            agent_c.append(species_color_palette_viz.get(agent_item_viz.species_name, "black"))

    if agent_x:
        plt.scatter(agent_x, agent_y, c=agent_c, s=10, marker="o", alpha=0.7, edgecolor='gray', linewidth=0.3)

    # Add island names from model.islands_info
    if hasattr(model, 'islands_info'):
        for island_data_viz in model.islands_info:
            if island_data_viz["area_cells"] > 0: # Only label islands that are on grid
                plt.text(island_data_viz["centroid_grid"][0], island_data_viz["centroid_grid"][1], 
                         island_data_viz["name"], ha='center', va='center', color='black', fontsize=7,
                         bbox=dict(facecolor='white', alpha=0.5, pad=0.2, boxstyle='round,pad=0.1'))

    plt.title(f"Agent Distribution at Step {model.schedule.steps}", fontsize=14)
    plt.xlabel("X Coordinate", fontsize=10)
    plt.ylabel("Y Coordinate", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.2)
    plt.xlim(0, model.width)
    plt.ylim(0, model.height)
    # plt.gca().set_aspect('equal', adjustable='box') # Can make it too small if grid is not square
    plt.tight_layout()
    plt.show()


# --- Simulation Run & Visualization ---
if __name__ == "__main__":
    print("Starting Galapagos ABM Simulation...")
    
    # IMPORTANT: Create a dummy GIS file for testing if you don't have one.
    # For example, a simple GeoJSON with a few polygons.
    # If GALAPAGOS_GIS_FILE is not found, the simulation will run with no islands.
    # You can create a dummy_galapagos.geojson like this:
    # {
    #   "type": "FeatureCollection",
    #   "features": [
    #     { "type": "Feature", "properties": { "NAME_2": "IslaDummy1" }, "geometry": { "type": "Polygon", "coordinates": [ [ [10,10], [10,30], [30,30], [30,10], [10,10] ] ] } },
    #     { "type": "Feature", "properties": { "NAME_2": "IslaDummy2" }, "geometry": { "type": "Polygon", "coordinates": [ [ [50,50], [50,70], [70,70], [70,50], [50,50] ] ] } }
    #   ]
    # }
    # And place it at the path specified in GALAPAGOS_GIS_FILE.
    # Note: The coordinates in the dummy file are relative to its own system.
    # The model will scale them to fit the GRID_WIDTH/HEIGHT.

    model_instance = GalapagosModel(
        GRID_WIDTH, GRID_HEIGHT, 
        GALAPAGOS_GIS_FILE, ISLAND_NAME_COLUMN_IN_GIS,
        SPECIES_DATA, 
        INITIAL_POP_PER_SPECIES_PER_ISLAND, 
        MAINLAND_IMMIGRATION_RATE,
        MAINLAND_POINT
    )

    if not model_instance.islands_info:
        print("CRITICAL: No islands were loaded. The simulation might not produce meaningful ecological results.")
        print("Please check your GIS file path and content.")
    else:
        print(f"Model initialized with {len(model_instance.islands_info)} islands.")


    for i in range(SIMULATION_STEPS):
        model_instance.step()
        # Optional: add a condition to stop if no agents left
        if model_instance.schedule.get_agent_count() == 0 and i > 20: # Give some time for initial pop
             print(f"Stopping early at step {i+1} as all agents have perished.")
             break


    print("\nSimulation finished.")
    model_run_data = model_instance.datacollector.get_model_vars_dataframe()
    
    print("\n--- Simulation Results (Last 5 steps of collected data) ---")
    print(model_run_data.tail())

    # --- Standard Plots ---
    plt.figure(figsize=(12, 6))
    for species_name_plot_val in SPECIES_DATA.keys():
        total_col = f"{species_name_plot_val}_total"
        if total_col in model_run_data.columns:
            plt.plot(model_run_data.index, model_run_data[total_col], label=f"{species_name_plot_val} Total Pop")
    plt.xlabel("Step")
    plt.ylabel("Total Population")
    plt.title("Overall Species Populations Over Time")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot Island Richness (if islands exist)
    if model_instance.islands_info:
        plt.figure(figsize=(12, 6))
        for island_info_plot in model_instance.islands_info:
            richness_col_plot = f"Richness_{island_info_plot['name'].replace(' ', '_')}"
            if richness_col_plot in model_run_data.columns:
                plt.plot(model_run_data.index, model_run_data[richness_col_plot], label=f"{island_info_plot['name']} Richness")
        plt.xlabel("Step")
        plt.ylabel("Species Richness")
        plt.title("Island Species Richness Over Time")
        plt.legend(fontsize='small', ncol=2)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # --- Validation Plots ---
    if model_instance.islands_info and not model_run_data.empty:
        plot_species_area_relationship(model_run_data, model_instance.islands_info)
        plot_species_isolation_relationship(model_run_data, model_instance.islands_info, MAINLAND_POINT)
    else:
        print("Skipping SAR/SIR plots as no island data or model run data is available.")

    # --- Final Spatial Distribution ---
    plot_final_distribution(model_instance)

    print("\nTo run this, ensure you have a valid GIS file for Galapagos islands specified in GALAPAGOS_GIS_FILE.")
    print("A dummy GeoJSON example is provided in comments if you need to test the structure.")
