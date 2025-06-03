import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from main import (
    DEFAULTS, load_species_json, run_simulation,
    plot_species_area_relationship, plot_species_isolation_relationship,
    plot_final_distribution
)


st.set_page_config(page_title="Galápagos ABM", layout="wide")
st.title("🪶 Galápagos Island Biogeography – ABM Dashboard")


with st.sidebar:
    st.header("⚙️  Global settings")

    species_dict = load_species_json()

    # 2) grid & timing
    grid_w = st.number_input("Grid width",  100, 2000, DEFAULTS["GRID_WIDTH"])
    grid_h = st.number_input("Grid height", 100, 2000, DEFAULTS["GRID_HEIGHT"])
    steps_y = st.number_input("Steps / year", 4, 365,
                              DEFAULTS["STEPS_PER_YEAR"])
    sim_yrs = st.slider("Simulation years", 1, 20, DEFAULTS["SIM_YEARS"])
    immig = st.number_input("Annual mainland immigration", 0.0, 0.1,
                            DEFAULTS["ANNUAL_MAINLAND_IMMIGRATION_RATE"], 0.001)

    st.subheader("Initial population (default)")
    init_default = {}
    for sp in species_dict:
        init_default[sp] = st.number_input(
            f"{sp}", 0, 50, DEFAULTS["INITIAL_POP_DEFAULT"])

    st.subheader("Include / exclude species")
    enabled = {sp: st.checkbox(sp, True) for sp in species_dict}

    per_island = {}
    if st.checkbox("✏️  advanced: customise per-island start numbers"):
        # we need island ids ⇒ spin up tiny model just to read gis
        from galapagos_model import GalapagosModel, normalize_simulation_data, GALAPAGOS_GIS_FILE, ISLAND_NAME_COLUMN_IN_GIS
        tiny_proc, _ = normalize_simulation_data(species_dict, immig, steps_y)
        tiny = GalapagosModel(200, 200, GALAPAGOS_GIS_FILE, ISLAND_NAME_COLUMN_IN_GIS,
                              tiny_proc, 0, 0.0, (0, 0))
        isl_ids = {i["id"]: i["name"] for i in tiny.islands_info}

        tbl = pd.DataFrame(0, index=list(species_dict), columns=isl_ids.keys())
        tbl.columns = [isl_ids[c] for c in tbl.columns]  # pretty names
        tbl_key = "pop_table"
        tbl = st.data_editor(tbl, num_rows="dynamic", key=tbl_key)

        # convert back to dict[species][island_id]=n
        per_island = {
            sp: {iid: int(tbl.loc[sp, isl_ids[iid]]) for iid in isl_ids}
            for sp in tbl.index
        }

    run_btn = st.button("🚀  Run simulation")


if run_btn:
    with st.spinner("Running …"):
        settings = dict(
            GRID_WIDTH=grid_w,
            GRID_HEIGHT=grid_h,
            STEPS_PER_YEAR=steps_y,
            SIM_YEARS=sim_yrs,
            INITIAL_POP_DEFAULT=max(init_default.values()),
            ANNUAL_MAINLAND_IMMIGRATION_RATE=immig,
        )
        # keep only enabled species
        sp_filtered = {k: v for k, v in species_dict.items()
                       if enabled.get(k, False)}
        model, df = run_simulation(settings, sp_filtered, per_island)

    st.success("Done!")

    st.subheader("Populations per species")
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in df.columns:
        if col.endswith("_total"):
            ax.plot(df.index, df[col], label=col.replace("_total", ""))
    ax.set_xlabel("Week")
    ax.set_ylabel("Population")
    ax.legend()
    ax.grid(alpha=.3, linestyle=":")
    st.pyplot(fig, use_container_width=False)

    if model.islands_info:
        st.subheader("Species richness (last 50 weeks avg.)")
        last50 = df.tail(min(50, len(df)))
        rich = {isl['name']: last50[f"Richness_{isl['name'].replace(' ','_')}"].mean()
                for isl in model.islands_info
                if f"Richness_{isl['name'].replace(' ','_')}" in last50.columns}
        st.bar_chart(pd.Series(rich))

    st.subheader("Final distribution")
    with plt.rc_context({'figure.figsize': (8, 8)}):
        plot_final_distribution(model)
        st.pyplot(plt.gcf(), use_container_width=False)

    st.subheader("SAR / SIR")
    with plt.rc_context({'figure.figsize': (7, 4)}):
        plot_species_area_relationship(df, model.islands_info)
        st.pyplot(plt.gcf(), use_container_width=False)

    with plt.rc_context({'figure.figsize': (7, 4)}):
        plot_species_isolation_relationship(
            df, model.islands_info, model.mainland_point)
        st.pyplot(plt.gcf(), use_container_width=False)

else:
    st.info("Configure parameters on the left & click **Run simulation**.")
