import json
from typing import Any
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from main import (
    DEFAULTS,
    load_species_json,
    run_simulation,
    plot_species_area_relationship,
    plot_species_isolation_relationship,
    plot_final_distribution,
)


st.set_page_config(page_title="Galápagos ABM", layout="wide")
st.title("🪶 Galápagos Island Biogeography – ABM Dashboard")


with st.sidebar:
    st.header("⚙️  Global settings")

    species_dict = load_species_json()

    # 2) grid & timing
    grid_w = st.number_input("Grid width", 100, 2000,
                             DEFAULTS["GRID_WIDTH"], step=50)
    grid_h = st.number_input("Grid height", 100, 2000,
                             DEFAULTS["GRID_HEIGHT"], step=50)
    steps_y = st.number_input("Steps / year", 4, 365,
                              DEFAULTS["STEPS_PER_YEAR"], step=1)
    sim_yrs = st.slider("Simulation years", 1, 20, DEFAULTS["SIM_YEARS"])
    immig = st.number_input(
        "Annual mainland immigration",
        min_value=0.0,
        max_value=0.1,
        value=DEFAULTS["ANNUAL_MAINLAND_IMMIGRATION_RATE"],
        step=0.001,
        format="%.3f",
    )

    st.subheader("Initial population (default)")
    init_default: dict[str, int] = {}
    for sp in species_dict:
        init_default[sp] = st.number_input(
            f"{sp}", min_value=0, max_value=50000, value=DEFAULTS["INITIAL_POP_DEFAULT"], step=1
        )

    st.subheader("Include / exclude species")
    enabled: dict[str, bool] = {
        sp: st.checkbox(sp, True) for sp in species_dict}

    per_island: dict[str, dict[int, int]] = {}
    if st.checkbox("✏️  advanced: customise per-island start numbers"):
        from main import GalapagosModel, normalize_simulation_data, GALAPAGOS_GIS_FILE, ISLAND_NAME_COLUMN_IN_GIS

        # Build a tiny model to fetch island IDs
        tiny_proc, _ = normalize_simulation_data(species_dict, immig, steps_y)
        tiny = GalapagosModel(
            width=200,
            height=200,
            gis_file=GALAPAGOS_GIS_FILE,
            island_name_col=ISLAND_NAME_COLUMN_IN_GIS,
            processed_species_data=tiny_proc,
            default_init_pop=0,
            mainland_immigration_rate_per_step=0.0,
            mainland_point=(0, 0),
        )
        isl_ids = {i["id"]: i["name"] for i in tiny.islands_info}

        tbl = pd.DataFrame(0, index=list(species_dict), columns=isl_ids.keys())
        tbl.columns = [isl_ids[c] for c in tbl.columns]  # pretty names
        tbl_key = "pop_table"
        tbl = st.data_editor(tbl, num_rows="dynamic", key=tbl_key)

        # Convert back to dict[species][island_id] = n
        per_island = {
            sp: {iid: int(tbl.loc[sp, isl_ids[iid]]) for iid in isl_ids}
            for sp in tbl.index
        }

    run_btn = st.button("🚀  Run simulation")


if run_btn:

    if not per_island:
        from main import GalapagosModel, normalize_simulation_data, GALAPAGOS_GIS_FILE, ISLAND_NAME_COLUMN_IN_GIS
        tiny_proc, _ = normalize_simulation_data(species_dict, immig, steps_y)
        tiny = GalapagosModel(
            width=200,
            height=200,
            gis_file=GALAPAGOS_GIS_FILE,
            island_name_col=ISLAND_NAME_COLUMN_IN_GIS,
            processed_species_data=tiny_proc,
            default_init_pop=0,
            mainland_immigration_rate_per_step=0.0,
            mainland_point=(0, 0),
        )
        isl_ids = [isl["id"] for isl in tiny.islands_info]

        per_island = {
            sp: {iid: init_default[sp] for iid in isl_ids}
            for sp in species_dict
        }

    with st.spinner("Running … this can take 30–60 seconds …"):
        settings = dict(
            GRID_WIDTH=grid_w,
            GRID_HEIGHT=grid_h,
            STEPS_PER_YEAR=steps_y,
            SIM_YEARS=sim_yrs,
            INITIAL_POP_DEFAULT=0,
            ANNUAL_MAINLAND_IMMIGRATION_RATE=immig,
        )
        sp_filtered = {k: v for k, v in species_dict.items() if enabled[k]}
        model, df = run_simulation(settings, sp_filtered, per_island)

    st.success("✅ Simulation finished!")

    # 1) Populations per species
    st.subheader("Populations per species (weekly)")
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in df.columns:
        if col.endswith("_total"):
            ax.plot(df.index, df[col], label=col.replace("_total", ""))
    ax.set_xlabel("Week")
    ax.set_ylabel("Population")
    ax.legend()
    ax.grid(alpha=.3, linestyle=":")
    st.pyplot(fig, use_container_width=False)

    # 2) Species richness (last 50 weeks avg)
    if model.islands_info:
        st.subheader("Species richness (last 50 weeks avg.)")
        last50 = df.tail(min(50, len(df)))
        rich = {
            isl["name"]: last50[f"Richness_{isl['name'].replace(' ', '_')}"].mean(
            )
            for isl in model.islands_info
            if f"Richness_{isl['name'].replace(' ', '_')}" in last50.columns
        }
        st.bar_chart(pd.Series(rich))

    # 3) Final distribution
    st.subheader("Final distribution")
    with plt.rc_context({"figure.figsize": (8, 8)}):
        plot_final_distribution(model)
        st.pyplot(plt.gcf(), use_container_width=False)

    # 4) SAR / SIR
    st.subheader("Species–Area Relationship (SAR)")
    with plt.rc_context({"figure.figsize": (7, 4)}):
        plot_species_area_relationship(df, model.islands_info)
        st.pyplot(plt.gcf(), use_container_width=False)

    st.subheader("Species–Isolation Relationship (SIR)")
    with plt.rc_context({"figure.figsize": (7, 4)}):
        plot_species_isolation_relationship(
            df, model.islands_info, model.mainland_point)
        st.pyplot(plt.gcf(), use_container_width=False)

    # ──────────────────────────────────────────────────────────────────
    # 5) Per‐island × Per‐species (final week) table
    # ──────────────────────────────────────────────────────────────────
    st.subheader("Per‐island × per‐species populations (final week)")
    pop_cols = [c for c in df.columns if c.startswith("Pop_")]
    if pop_cols:
        final_row = df.iloc[-1]
        records: list[dict[str, Any]] = []
        for col in pop_cols:
            # col format: "Pop_<IslandTag>_<Species>"
            parts = col.split("_", maxsplit=2)
            island_tag = parts[1]
            species_name = parts[2]
            count_val = int(final_row[col])
            records.append(
                {"Island": island_tag, "Species": species_name, "Count": count_val})

        table_df = pd.DataFrame.from_records(records)
        pivot_df = table_df.pivot(
            index="Island", columns="Species", values="Count").fillna(0).astype(int)
        st.dataframe(pivot_df, use_container_width=False)
    else:
        st.info("No per-island data was recorded. Check that `_setup_datacollector` includes Pop_<Island>_<Species> reporters.")

else:
    st.info("Configure parameters on the left & click **Run simulation**.")
