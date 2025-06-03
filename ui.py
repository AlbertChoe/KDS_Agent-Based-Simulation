# Simple Streamlit interface for the Galápagos ABM
# -------------------------------------------------
# Drop this file (galapagos_abm_ui.py) alongside your existing model
# code (the long script you shared). Make sure that script is in the
# Python path as a module, e.g. save it as `galapagos_model.py` or
# similar so we can `import` from it.
#
# Usage:
#   streamlit run galapagos_abm_ui.py
# -------------------------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from main import plot_species_area_relationship, plot_species_isolation_relationship

# Try to import the model (rename the big script to galapagos_model.py)
try:
    from main import (
        GalapagosModel,
        normalize_simulation_data,
        SPECIES_DATA_ANNUAL,
        GRID_WIDTH,
        GRID_HEIGHT,
        GALAPAGOS_GIS_FILE,
        ISLAND_NAME_COLUMN_IN_GIS,
        MAINLAND_POINT,
    )
except ModuleNotFoundError:
    st.error("Could not import the model module …")
    st.stop()

# ---------------------------------------
# Sidebar – simulation controls
# ---------------------------------------

st.sidebar.title("⚙️ Simulation Controls")

# Years and weekly steps
steps_per_year = 52
sim_years = st.sidebar.slider("Simulation duration (years)", 1, 10, 3)
steps_total = steps_per_year * sim_years

# Immigration rate tweak
annual_immigration = st.sidebar.number_input(
    "Annual mainland immigration rate",
    min_value=0.0,
    max_value=0.05,
    value=0.005,
    step=0.001,
    format="%0.3f",
)

# Initial population per species
initial_pop = st.sidebar.number_input(
    "Initial pop per species / island",
    min_value=1,
    max_value=20,
    value=3,
    step=1,
)

# Species toggle
st.sidebar.markdown("### Include species")
species_enabled = {}
for sp in SPECIES_DATA_ANNUAL.keys():
    species_enabled[sp] = st.sidebar.checkbox(sp, value=True)

# Run button
run_button = st.sidebar.button("🚀 Run Simulation")

# ---------------------------------------
# Main UI
# ---------------------------------------
st.title("Galápagos Island Biogeography – ABM Dashboard")

st.markdown(
    "This simple interface lets you tweak a few high‑level parameters and **run the agent‑based\n     model live**. It will display population trajectories and the final spatial distribution.\n     For full parameter control, edit `SPECIES_DATA_ANNUAL` in code."
)

# ---------------------------------------
# Helper – run model & collect results
# ---------------------------------------


def run_model(years: int, annual_imm_rate: float, init_pop: int):
    # 1) filter species dict
    annual_dict = {k: v for k, v in SPECIES_DATA_ANNUAL.items()
                   if species_enabled.get(k, False)}

    # 2) normalise to weekly values
    processed_dict, imm_rate_per_step = normalize_simulation_data(
        annual_dict, annual_imm_rate, steps_per_year
    )

    # 3) instantiate model (smaller grid 400×400 for speed)
    model = GalapagosModel(
        width=400,
        height=400,
        gis_file=GALAPAGOS_GIS_FILE,
        island_name_col=ISLAND_NAME_COLUMN_IN_GIS,
        processed_species_data=processed_dict,
        initial_pop=init_pop,
        mainland_immigration_rate_per_step=imm_rate_per_step,
        mainland_point=MAINLAND_POINT,
    )

    for step in range(steps_per_year * years):
        model.step()
        # quick escape if extinct
        if model.schedule.get_agent_count() == 0:
            break

    df = model.datacollector.get_model_vars_dataframe()
    return model, df


# ---------------------------------------
# Trigger simulation
# ---------------------------------------
if run_button:
    with st.spinner("Running simulation … this may take up to a minute …"):
        model_obj, results_df = run_model(
            sim_years, annual_immigration, initial_pop)

    st.success("Simulation finished!")

    # Plot total populations
    st.subheader("Total populations per species (weekly)")
    fig, ax = plt.subplots(figsize=(10, 4))
    for sp in results_df.columns:
        if sp.endswith("_total"):
            ax.plot(results_df.index,
                    results_df[sp], label=sp.replace("_total", ""))
    ax.set_xlabel("Week")
    ax.set_ylabel("Population")
    ax.legend()
    ax.grid(alpha=0.3, linestyle=":")
    st.pyplot(fig)

    # Plot island richness if islands exist
    if model_obj.islands_info:
        st.subheader("Island species richness (last 50 weeks average)")
        last50 = results_df.tail(min(50, len(results_df)))
        isl_names, richness_vals = [], []
        for isl in model_obj.islands_info:
            col = f"Richness_{isl['name'].replace(' ', '_')}"
            if col in last50.columns:
                isl_names.append(isl['name'])
                richness_vals.append(last50[col].mean())
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(isl_names, richness_vals)
        ax2.set_ylabel("Mean richness (last 50 weeks)")
        ax2.set_xticks(range(len(isl_names)))
        ax2.set_xticklabels(isl_names, rotation=45, ha="right")
        st.pyplot(fig2)

    # Final spatial distribution (static png)
    # Final spatial distribution
    # Final spatial distribution
    st.subheader("Final spatial distribution")
    try:
        from main import plot_final_distribution

        # Close any leftover figures
        plt.close('all')

        # plot_final_distribution will create a new figure and draw onto it
        plot_final_distribution(model_obj)

        # Streamlit grabs the active figure
        st.pyplot(plt.gcf())

    except Exception as e:
        st.warning("Could not draw distribution plot – " + str(e))

    st.subheader("Species–Area Relationship (SAR)")
    try:
        plt.close('all')
        plot_species_area_relationship(results_df, model_obj.islands_info)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning("Could not draw SAR plot – " + str(e))

    # Species–Isolation Relationship (SIR)
    st.subheader("Species–Isolation Relationship (SIR)")
    try:
        plt.close('all')
        plot_species_isolation_relationship(
            results_df, model_obj.islands_info, MAINLAND_POINT)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning("Could not draw SIR plot – " + str(e))
else:
    st.info("Use the sidebar to configure parameters, then press **Run Simulation**.")
