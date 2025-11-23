
# # # import streamlit as st
# # # import numpy as np
# # # import time

# # # st.set_page_config(page_title="UCB Ad Selection Visual Simulation", layout="wide")
# # # st.title("🎯 Upper Confidence Bound (UCB) Ad Selection Simulation")

# # # # --- Parameters ---
# # # true_ctrs = [0.05, 0.10, 0.20]
# # # n_rounds = st.slider("Number of user visits (rounds)", 100, 20000, 200)
# # # c = st.slider("Exploration parameter (c)", 0.0, 100.0, 2.0)
# # # speed = st.slider("Animation speed (seconds per step)", 0.0001, 0.5, 0.1)

# # # # --- Initialize state ---
# # # if "Q" not in st.session_state:
# # #     st.session_state.Q = np.zeros(len(true_ctrs))
# # #     st.session_state.N = np.zeros(len(true_ctrs))
# # #     st.session_state.rewards = []
# # #     st.session_state.simulation_running = False

# # # # --- Layout ---
# # # col1, col2 = st.columns([2, 1])

# # # with col1:
# # #     st.subheader("🖥️ Website Ad Display Simulation")
# # #     ad_placeholders = [st.empty() for _ in range(len(true_ctrs))]
# # #     st.markdown("---")

# # # with col2:
# # #     st.subheader("📊 Statistics")
# # #     stats = st.empty()

# # # # --- Start button ---
# # # start_button = st.button("🚀 Start Simulation")

# # # if start_button:
# # #     # Reset state before running
# # #     st.session_state.Q = np.zeros(len(true_ctrs))
# # #     st.session_state.N = np.zeros(len(true_ctrs))
# # #     st.session_state.rewards = []
# # #     st.session_state.simulation_running = True

# # # # --- Simulation ---
# # # if st.session_state.simulation_running:
# # #     progress = st.progress(0)

# # #     for t in range(1, n_rounds + 1):
# # #         Q, N = st.session_state.Q, st.session_state.N
# # #         ucb_values = np.zeros(len(true_ctrs))

# # #         for a in range(len(true_ctrs)):
# # #             if N[a] == 0:
# # #                 ucb_values[a] = float('inf')
# # #             else:
# # #                 ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])

# # #         chosen_ad = np.argmax(ucb_values)
# # #         reward = np.random.rand() < true_ctrs[chosen_ad]
# # #         st.session_state.rewards.append(int(reward))

# # #         # Update estimates
# # #         N[chosen_ad] += 1
# # #         Q[chosen_ad] += (reward - Q[chosen_ad]) / N[chosen_ad]

# # #         # --- Visual display ---
# # #         for i in range(len(true_ctrs)):
# # #             if i == chosen_ad:
# # #                 ad_placeholders[i].markdown(
# # #                     f"""
# # #                     <div style='border:3px solid green;padding:10px;border-radius:10px;text-align:center;background:#f0fff0;'>
# # #                         <h4>🟩 Ad {i+1}</h4>
# # #                         <p>Displayed ✅</p>
# # #                     </div>
# # #                     """,
# # #                     unsafe_allow_html=True,
# # #                 )
# # #             else:
# # #                 ad_placeholders[i].markdown(
# # #                     f"""
# # #                     <div style='border:1px solid #ccc;padding:10px;border-radius:10px;text-align:center;background:#fafafa;'>
# # #                         <h4>⬜ Ad {i+1}</h4>
# # #                         <p>Idle</p>
# # #                     </div>
# # #                     """,
# # #                     unsafe_allow_html=True,
# # #                 )

# # #         stats.markdown(
# # #             f"""
# # #             **Round:** {t}/{n_rounds}  
# # #             **Total Clicks:** {int(sum(st.session_state.rewards))}  
# # #             **Chosen Ad:** Ad {chosen_ad+1} {'✅ Click!' if reward else '❌ No Click'}  
# # #             ---
# # #             **Estimated CTRs:** {np.round(Q, 3).tolist()}  
# # #             **True CTRs:** {true_ctrs}  
# # #             **Selection Counts:** {N.astype(int).tolist()} 
# # #             **UBC:** {ucb_values.astype(float).tolist()}
# # #             """
# # #         )

# # #         progress.progress(t / n_rounds)
# # #         time.sleep(speed)

# # #     st.success("✅ Simulation complete!")
# # #     st.session_state.simulation_running = False


# # # import streamlit as st
# # # import numpy as np
# # # import time
# # # import pandas as pd

# # # st.set_page_config(page_title="UCB Ad Selection Visual + Comparison", layout="wide")
# # # st.title("🎯 Upper Confidence Bound (UCB) Ad Selection – Visual + Multi-C Analysis")

# # # # --- Parameters ---
# # # true_ctrs = [0.05, 0.10, 0.20]
# # # n_rounds = st.slider("Number of user visits (rounds)", 100, 2000, 500)
# # # c_values = st.multiselect("Exploration parameter values (c)", [0.0,  1.0, 10, 100,1000,10000], default=[1.0])
# # # speed = st.slider("Animation speed (seconds per step for visual sim)", 0.01, 0.5, 0.05)

# # # # --- Buttons ---
# # # colA, colB = st.columns([1, 1])
# # # with colA:
# # #     run_visual = st.button("🚀 Run Visual Simulation (first c value only)")
# # # with colB:
# # #     run_multi = st.button("📈 Run Comparison for all c values")

# # # # --- Helper function ---
# # # def run_ucb_simulation(c, n_rounds, true_ctrs):
# # #     n_ads = len(true_ctrs)
# # #     Q = np.zeros(n_ads)
# # #     N = np.zeros(n_ads)
# # #     total_reward = 0
# # #     rewards_over_time = []

# # #     for t in range(1, n_rounds + 1):
# # #         ucb_values = np.zeros(n_ads)
# # #         for a in range(n_ads):
# # #             if N[a] == 0:
# # #                 ucb_values[a] = float('inf')
# # #             else:
# # #                 ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])
# # #         chosen_ad = np.argmax(ucb_values)
# # #         reward = np.random.rand() < true_ctrs[chosen_ad]
# # #         total_reward += int(reward)
# # #         rewards_over_time.append(total_reward)
# # #         N[chosen_ad] += 1
# # #         Q[chosen_ad] += (reward - Q[chosen_ad]) / N[chosen_ad]
# # #     return np.array(rewards_over_time)

# # # # --- VISUAL SIMULATION for 1 value of c ---
# # # if run_visual:
# # #     c = c_values[0]
# # #     st.subheader(f"🎬 Visual Simulation for c = {c}")
# # #     col1, col2 = st.columns([2, 1])

# # #     with col1:
# # #         ad_placeholders = [st.empty() for _ in range(len(true_ctrs))]
# # #         progress = st.progress(0)

# # #     with col2:
# # #         stats = st.empty()

# # #     Q = np.zeros(len(true_ctrs))
# # #     N = np.zeros(len(true_ctrs))
# # #     rewards = []

# # #     for t in range(1, n_rounds + 1):
# # #         ucb_values = np.zeros(len(true_ctrs))
# # #         for a in range(len(true_ctrs)):
# # #             if N[a] == 0:
# # #                 ucb_values[a] = float('inf')
# # #             else:
# # #                 ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])
# # #         chosen_ad = np.argmax(ucb_values)
# # #         reward = np.random.rand() < true_ctrs[chosen_ad]
# # #         rewards.append(int(reward))
# # #         N[chosen_ad] += 1
# # #         Q[chosen_ad] += (reward - Q[chosen_ad]) / N[chosen_ad]

# # #         # Visualization
# # #         for i in range(len(true_ctrs)):
# # #             if i == chosen_ad:
# # #                 ad_placeholders[i].markdown(
# # #                     f"<div style='border:3px solid green;padding:10px;border-radius:10px;text-align:center;background:#f0fff0;'>"
# # #                     f"<h4>🟩 Ad {i+1}</h4><p>Displayed ✅</p></div>",
# # #                     unsafe_allow_html=True,
# # #                 )
# # #             else:
# # #                 ad_placeholders[i].markdown(
# # #                     f"<div style='border:1px solid #ccc;padding:10px;border-radius:10px;text-align:center;background:#fafafa;'>"
# # #                     f"<h4>⬜ Ad {i+1}</h4><p>Idle</p></div>",
# # #                     unsafe_allow_html=True,
# # #                 )
# # #         stats.markdown(
# # #             f"**Round:** {t}/{n_rounds}  \n"
# # #             f"**Total Clicks:** {int(sum(rewards))}  \n"
# # #             f"**Chosen Ad:** Ad {chosen_ad+1} {'✅ Click!' if reward else '❌ No Click'}  \n"
# # #             f"---\n"
# # #             f"**Estimated CTRs:** {np.round(Q, 3).tolist()}  \n"
# # #             f"**True CTRs:** {true_ctrs}  \n"
# # #             f"**Selection Counts:** {N.astype(int).tolist()}"
# # #         )
# # #         progress.progress(t / n_rounds)
# # #         time.sleep(speed)
# # #     st.success("✅ Simulation complete!")

# # # # --- MULTI-C COMPARISON ---
# # # if run_multi:
# # #     st.subheader("📊 Total Reward vs. Rounds for Different c Values")
# # #     results = {}
# # #     for c in c_values:
# # #         rewards = run_ucb_simulation(c, n_rounds, true_ctrs)
# # #         results[f"c={c}"] = rewards
# # #     df = pd.DataFrame(results)
# # #     st.line_chart(df)
# # #     st.markdown("The steeper the curve, the faster the agent learns and collects reward. Low c → exploit more, High c → explore more.")


# # # import streamlit as st
# # # import numpy as np
# # # import pandas as pd
# # # from scipy import stats

# # # st.set_page_config(page_title="UCB Ad Selection – Statistical Analysis", layout="wide")
# # # st.title("🎯 UCB Ad Selection — Statistical Test for Best Exploration Parameter (c)")

# # # # --- Parameters ---
# # # true_ctrs = [0.05, 0.10, 0.20]
# # # n_rounds = st.slider("Rounds per simulation", 200, 2000, 500)
# # # n_trials = st.slider("Simulations per c value", 10, 100, 30)
# # # c_values = st.multiselect("Exploration parameters (c) to test", [0.1, 0.5, 1.0, 1.5, 2.0, 3.0], default=[0.5, 1.0, 2.0])
# # # run_analysis = st.button("📊 Run Statistical Evaluation")

# # # # --- Simulation Function ---
# # # def run_ucb_once(c, n_rounds, true_ctrs):
# # #     n_ads = len(true_ctrs)
# # #     Q = np.zeros(n_ads)
# # #     N = np.zeros(n_ads)
# # #     total_reward = 0
# # #     for t in range(1, n_rounds + 1):
# # #         ucb_values = np.zeros(n_ads)
# # #         for a in range(n_ads):
# # #             if N[a] == 0:
# # #                 ucb_values[a] = float('inf')
# # #             else:
# # #                 ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])
# # #         chosen_ad = np.argmax(ucb_values)
# # #         reward = np.random.rand() < true_ctrs[chosen_ad]
# # #         total_reward += int(reward)
# # #         N[chosen_ad] += 1
# # #         Q[chosen_ad] += (reward - Q[chosen_ad]) / N[chosen_ad]
# # #     return total_reward

# # # # --- Run statistical evaluation ---
# # # if run_analysis:
# # #     st.info("Running multiple simulations... this may take a few seconds ⏳")

# # #     results = {c: [] for c in c_values}
# # #     for c in c_values:
# # #         for trial in range(n_trials):
# # #             total = run_ucb_once(c, n_rounds, true_ctrs)
# # #             results[c].append(total)

# # #     df = pd.DataFrame(dict([(f"c={c}", results[c]) for c in c_values]))

# # #     # --- Show descriptive stats ---
# # #     st.subheader("📈 Summary Statistics")
# # #     summary = df.describe().T[["mean", "std"]]
# # #     summary["mean ± std"] = summary["mean"].round(2).astype(str) + " ± " + summary["std"].round(2).astype(str)
# # #     st.dataframe(summary[["mean ± std"]])

# # #     # --- ANOVA test ---
# # #     st.subheader("📊 ANOVA Significance Test")
# # #     anova_stat, p_value = stats.f_oneway(*[results[c] for c in c_values])
# # #     st.write(f"**F-statistic:** {anova_stat:.3f}, **p-value:** {p_value:.4f}")

# # #     if p_value < 0.05:
# # #         st.success("✅ There is a statistically significant difference between at least two c values.")
# # #     else:
# # #         st.warning("⚠️ No significant difference found (p ≥ 0.05).")

# # #     # --- Highlight best mean ---
# # #     best_c = summary["mean"].idxmax()
# # #     st.subheader(f"🏆 Best performing c value: **{best_c}** (Highest mean reward)")

# # #     # --- Plot distributions ---
# # #     st.subheader("🎨 Reward Distribution per c Value")
# # #     st.box_chart(df)


# # # import streamlit as st
# # # import numpy as np
# # # import pandas as pd
# # # import altair as alt
# # # from scipy import stats
# # # import itertools
# # # import time

# # # # --- Page setup ---
# # # st.set_page_config(page_title="UCB Ad Simulation with Statistical Testing", layout="wide")
# # # st.title("🎯 Upper Confidence Bound (UCB) Ad Selection Simulation with Statistical Analysis")

# # # # --- Parameters ---
# # # true_ctrs = [0.05, 0.10, 0.20]
# # # st.sidebar.header("⚙️ Simulation Settings")

# # # n_rounds = st.sidebar.slider("Number of rounds (user visits)", 100, 2000, 500)
# # # speed = st.sidebar.slider("Animation speed (seconds per step)", 0.01, 0.5, 0.05)
# # # c_values = st.sidebar.text_input("Exploration parameter values (comma-separated)", "0.5,1.0,2.0,3.0")
# # # n_runs = st.sidebar.slider("Number of runs per c", 1, 30, 10)

# # # c_values = [float(x.strip()) for x in c_values.split(",") if x.strip()]

# # # # --- Initialize Session State ---
# # # if "results" not in st.session_state:
# # #     st.session_state.results = {}

# # # # --- Start Simulation Button ---
# # # start = st.button("🚀 Run Simulation")

# # # if start:
# # #     st.session_state.results = {} # {}
# # #     progress = st.progress(0)
# # #     status = st.empty()

# # #     for ci, c in enumerate(c_values): # {0,0.5,1,2,3,4}
# # #         total_rewards = []
# # #         status.info(f"Running simulations for **c = {c}** ...")

# # #         for run in range(n_runs): # n_runs = 30
# # #             Q = np.zeros(len(true_ctrs)) # {0,0,0}
# # #             N = np.zeros(len(true_ctrs)) # {0,0,0}
# # #             rewards = [] #[]

# # #             for t in range(1, n_rounds + 1):
# # #                 ucb_values = np.zeros(len(true_ctrs)) #{inf,inf,inf}
# # #                 for a in range(len(true_ctrs)): 
# # #                     if N[a] == 0:
# # #                         ucb_values[a] = float("inf")
# # #                     else:
# # #                         ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])

# # #                 chosen_ad = np.argmax(ucb_values)
# # #                 reward = np.random.rand() < true_ctrs[chosen_ad]
# # #                 rewards.append(int(reward))

# # #                 N[chosen_ad] += 1
# # #                 Q[chosen_ad] += (reward - Q[chosen_ad]) / N[chosen_ad]

# # #             #after all rounds are complete
# # #             total_rewards.append(sum(rewards)/n_rounds)

# # #         st.session_state.results[c] = total_rewards
# # #         progress.progress((ci + 1) / len(c_values))

# # #     progress.progress(1.0)
# # #     status.success("✅ All simulations completed!")

# # # # --- Display Results ---
# # # if st.session_state.results:
# # #     st.header("📊 Results Analysis")

# # #     df = pd.DataFrame(st.session_state.results)

# # #     st.subheader("📦 Raw Reward Data")
# # #     st.dataframe(df)

# # #     # --- Box Plot ---
# # #     st.subheader("🎨 Reward Distribution per c Value")
# # #     df_melted = df.melt(var_name="c_value", value_name="total_reward")

# # #     box_chart = (
# # #         alt.Chart(df_melted)
# # #         .mark_boxplot(size=40)
# # #         .encode(
# # #             x=alt.X("c_value:N", title="Exploration parameter (c)"),
# # #             y=alt.Y("total_reward:Q", title="Total Reward per Simulation"),
# # #             color=alt.Color("c_value:N", legend=None),
# # #         )
# # #         .properties(width=600, height=400)
# # #     )
# # #     st.altair_chart(box_chart, use_container_width=True)

# # #     # --- Mean Reward Trend (with individual runs connected) ---
# # #     st.subheader("📈 Average CTR vs. c Value (All Runs + Mean Trend)")

# # #     # Melt dataframe for Altair
# # #     df_melted = df.melt(var_name="c_value", value_name="avg_ctr")

# # #     # Ensure c_value is numeric and sorted ascending
# # #     df_melted["c_value"] = df_melted["c_value"].astype(float)
# # #     df_melted = df_melted.sort_values("c_value")

# # #     # Assign each simulation run an ID
# # #     df_melted["run_id"] = np.tile(np.arange(1, len(df) + 1), len(df.columns))

# # #     # Individual run traces
# # #     individual_lines = (
# # #         alt.Chart(df_melted)
# # #         .mark_line(opacity=0.3, color="gray")
# # #         .encode(
# # #             x=alt.X("c_value:Q", title="Exploration parameter (c)", sort="ascending"),
# # #             y=alt.Y("avg_ctr:Q", title="Average CTR"),
# # #             detail="run_id:N"
# # #         )
# # #     )

# # #     # Mean trend line
# # #     mean_line = (
# # #         alt.Chart(df_melted)
# # #         .mark_line(point=True, size=3, color="orange")
# # #         .encode(
# # #             x=alt.X("c_value:Q", title="Exploration parameter (c)", sort="ascending"),
# # #             y=alt.Y("mean(avg_ctr):Q", title="Mean Average CTR"),
# # #         )
# # #     )

# # #     # Combine and display
# # #     st.altair_chart(individual_lines + mean_line, use_container_width=True)
# # #     # --- Summary Stats ---
# # #     st.subheader("🧮 Statistical Summary")
# # #     summary = df.describe().T[["mean", "std", "min", "max"]]
# # #     st.dataframe(summary)

# # #     # --- Best c ---
# # #     best_c = summary["mean"].idxmax()
# # #     best_reward = summary.loc[best_c, "mean"]
# # #     st.success(f"🏆 Best c value: **{best_c}** with average total reward ≈ {best_reward:.1f}")

# # #     # --- Statistical Testing ---
# # #     st.subheader("🔬 Statistical Significance Tests")

# # #     # Perform one-way ANOVA
# # #     groups = [df[c] for c in df.columns]
# # #     f_stat, p_val = stats.f_oneway(*groups)
# # #     st.write(f"**One-way ANOVA** F = {f_stat:.4f}, p = {p_val:.4e}")

# # #     if p_val < 0.05:
# # #         st.success("✅ Significant difference found between c values (p < 0.05)")

# # #         # Pairwise t-tests with Bonferroni correction
# # #         pairs = list(itertools.combinations(df.columns, 2))
# # #         results = []
# # #         for (c1, c2) in pairs:
# # #             t_stat, p_pair = stats.ttest_ind(df[c1], df[c2])
# # #             results.append({
# # #                 "c1": c1,
# # #                 "c2": c2,
# # #                 "t_stat": t_stat,
# # #                 "p_raw": p_pair,
# # #                 "p_adj": min(p_pair * len(pairs), 1.0)  # Bonferroni correction
# # #             })

# # #         results_df = pd.DataFrame(results)
# # #         results_df["significant"] = results_df["p_adj"] < 0.05
# # #         st.dataframe(results_df)
# # #     else:
# # #         st.info("No statistically significant difference detected (p ≥ 0.05).")




# # # ucb_full_tutorial.py
# import streamlit as st
# import numpy as np
# import pandas as pd
# import altair as alt
# from scipy import stats
# import itertools
# import time

# st.set_page_config(page_title="UCB Tutorial & c-selection", layout="wide")
# st.title("🎯 UCB Tutorial — Visualize & Statistically Choose Best `c`")

# # -------------------------
# # Shared parameters & helpers
# # -------------------------
# DEFAULT_TRUE_CTRS = [0.05, 0.10, 0.20]

# def run_ucb_trace(c, n_rounds, true_ctrs, return_avg_ctr_over_time=False):
#     """
#     Run a single UCB simulation.
#     - If return_avg_ctr_over_time is True, returns array of average CTR (total clicks / rounds) at each round.
#     - Otherwise returns avg_ctr (mean clicks per impression) at the end.
#     """
#     n_ads = len(true_ctrs)
#     Q = np.zeros(n_ads)
#     N = np.zeros(n_ads)
#     clicks = np.zeros(n_rounds, dtype=int)

#     cumulative_clicks = 0
#     avg_ctr_over_time = np.zeros(n_rounds, dtype=float)

#     for t in range(1, n_rounds + 1):
#         ucb_values = np.zeros(n_ads)
#         for a in range(n_ads):
#             if N[a] == 0:
#                 ucb_values[a] = float("inf")
#             else:
#                 ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])

#         chosen = int(np.argmax(ucb_values))
#         reward = int(np.random.rand() < true_ctrs[chosen])
#         clicks[t - 1] = reward
#         cumulative_clicks += reward

#         # update
#         N[chosen] += 1
#         Q[chosen] += (reward - Q[chosen]) / N[chosen]

#         avg_ctr_over_time[t - 1] = cumulative_clicks / t

#     if return_avg_ctr_over_time:
#         return avg_ctr_over_time
#     else:
#         return clicks.mean()  # average CTR over impressions

# def run_ucb_once_return_clicks(c, n_rounds, true_ctrs):
#     # returns clicks array (0/1) for the run
#     n_ads = len(true_ctrs)
#     Q = np.zeros(n_ads)
#     N = np.zeros(n_ads)
#     clicks = np.zeros(n_rounds, dtype=int)
#     for t in range(1, n_rounds + 1):
#         ucb_values = np.zeros(n_ads)
#         for a in range(n_ads):
#             if N[a] == 0:
#                 ucb_values[a] = float("inf")
#             else:
#                 ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])
#         chosen = int(np.argmax(ucb_values))
#         reward = int(np.random.rand() < true_ctrs[chosen])
#         clicks[t - 1] = reward
#         N[chosen] += 1
#         Q[chosen] += (reward - Q[chosen]) / N[chosen]
#     return clicks

# # -------------------------
# # Sidebar controls
# # -------------------------
# st.sidebar.header("Global settings")
# true_ctrs = st.sidebar.text_input("True CTRs (comma-separated)", ",".join(map(str, DEFAULT_TRUE_CTRS)))
# try:
#     true_ctrs = [float(x.strip()) for x in true_ctrs.split(",") if x.strip() != ""]
#     if len(true_ctrs) < 2:
#         st.sidebar.error("Provide at least two CTRs (e.g. 0.05,0.10,0.20)")
# except Exception:
#     st.sidebar.error("Malformed CTRs. Use comma separated floats (e.g. 0.05,0.10,0.20)")
# n_rounds_global = st.sidebar.number_input("Default rounds (per run)", min_value=50, max_value=20000, value=500, step=50)
# st.sidebar.markdown("---")

# # -------------------------
# # Tabs: 1) Teach, 2) Single visual run, 3) Multi-c comparison, 4) Statistical selection
# # -------------------------
# tab1, tab2, tab3, tab4 = st.tabs(["1 — UCB primer", "2 — Visual single-run", "3 — Compare c over time", "4 — Statistical selection"])

# # -------------------------
# # Tab 1: Primer / Definitions
# # -------------------------
# with tab1:
#     st.header("1 — What are Q and R?")
#     st.markdown(
#         """
#         **Definitions**
#         - **Rₜ** — reward observed at time _t_: **1** if the user clicked the shown ad, **0** otherwise.
#         - **Qₜ(a)** — agent's estimate of the ad `a`'s expected reward (CTR) at time _t_.
        
#         **Update rule (sample-average incremental):**
#         \[
#         Q_{t+1}(a) = Q_t(a) + \frac{1}{N_t(a)} (R_t - Q_t(a))
#         \]
#         where \(N_t(a)\) is the number of times ad `a` has been shown so far.
        
#         This rule updates the running average of observed binary rewards — the mean of many 0/1 draws is the CTR (a probability).
#         """
#     )
#     st.write("**Quick numeric example (Ad 3, true CTR = 0.20)** — shows 10 impressions, clicks (Rₜ) and estimated Qₜ after each impression:")
#     example_clicks = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
#     Q = 0.0
#     table = []
#     for i, r in enumerate(example_clicks, start=1):
#         N = i
#         Q = Q + (r - Q) / N
#         table.append({"Impression": i, "R_t (click)": r, "Q_t (estimated CTR)": round(Q, 3)})
#     st.table(pd.DataFrame(table))

# # -------------------------
# # Tab 2: Visual single-run
# # -------------------------
# with tab2:
#     st.header("2 — Visual Single Run (step-by-step)")
#     st.markdown("Set parameters, click **Start single simulation** and watch how the agent explores and the estimated CTRs converge.")
#     col_left, col_right = st.columns([2, 1])

#     with col_right:
#         st.subheader("Settings")
#         n_rounds = st.number_input("Rounds (impressions)", min_value=50, max_value=10000, value=n_rounds_global)
#         c = st.number_input("Exploration parameter c", min_value=0.0, max_value=100.0, value=2.0, step=0.1, format="%.2f")
#         speed = st.slider("Animation speed (sleep seconds)", 0.0, 0.5, 0.02)
#         start_single = st.button("▶️ Start single simulation")
#         st.markdown("**True CTRs:** " + ", ".join([f"{x:.3f}" for x in true_ctrs]))

#     with col_left:
#         # placeholders for ad boxes and stats
#         ad_cols = st.columns(len(true_ctrs))
#         ad_placeholders = [col.empty() for col in ad_cols]
#         stats_placeholder = st.empty()
#         prog = st.progress(0.0)

#     if start_single:
#         # reset
#         Q = np.zeros(len(true_ctrs))
#         N = np.zeros(len(true_ctrs))
#         total_clicks = 0

#         for t in range(1, n_rounds + 1):
#             ucb_values = np.zeros(len(true_ctrs))
#             for a in range(len(true_ctrs)):
#                 if N[a] == 0:
#                     ucb_values[a] = float("inf")
#                 else:
#                     ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])

#             chosen = int(np.argmax(ucb_values))
#             reward = int(np.random.rand() < true_ctrs[chosen])
#             total_clicks += reward

#             # update
#             N[chosen] += 1
#             Q[chosen] += (reward - Q[chosen]) / N[chosen]

#             # render ad boxes
#             for i in range(len(true_ctrs)):
#                 if i == chosen:
#                     ad_placeholders[i].markdown(
#                         f"<div style='border:3px solid #2ecc71;padding:10px;border-radius:8px;text-align:center;background:#f0fff5;'>"
#                         f"<h4>🟩 Ad {i+1}</h4><p>Displayed — {'✅ Click' if reward else '❌ No click'}</p></div>",
#                         unsafe_allow_html=True,
#                     )
#                 else:
#                     ad_placeholders[i].markdown(
#                         f"<div style='border:1px solid #ddd;padding:10px;border-radius:8px;text-align:center;background:#fafafa;'>"
#                         f"<h4>⬜ Ad {i+1}</h4><p>Idle</p></div>",
#                         unsafe_allow_html=True,
#                     )

#             # show stats
#             stats_placeholder.markdown(
#                 f"""
#                 **Round:** {t}/{n_rounds}  
#                 **Avg CTR so far:** {(total_clicks / t):.4f}  
#                 **Estimated CTRs (Q):** {np.round(Q, 4).tolist()}  
#                 **Selections (N):** {N.astype(int).tolist()}  
#                 **UCB values:** {np.round(ucb_values,4).tolist()}
#                 """
#             )
#             prog.progress(t / n_rounds)
#             if speed > 0:
#                 time.sleep(speed)
#         st.success("Single simulation finished ✅")

# # -------------------------
# # Tab 3: Compare c over time (mean CTR curves)
# # -------------------------
# with tab3:
#     st.header("3 — Compare different `c` values over time")
#     st.markdown("Run several independent simulations per `c`, then plot average CTR vs rounds (learning curve) and per-run traces.")

#     colL, colR = st.columns([2, 1])
#     with colR:
#         n_rounds_comp = st.number_input("Rounds for comparison", min_value=50, max_value=5000, value=500, step=50)
#         c_list_input = st.text_input("c values (comma-separated)", "0.1,0.5,1.0,2.0,4.0")
#         n_runs_comp = st.number_input("Runs per c (for averaging)", min_value=1, max_value=50, value=8)
#         run_comp = st.button("📈 Run comparison")
#     try:
#         c_list = [float(x.strip()) for x in c_list_input.split(",") if x.strip() != ""]
#     except Exception:
#         st.error("Malformed c list")

#     with colL:
#         placeholder_chart = st.empty()
#         placeholder_mean = st.empty()

#     if run_comp:
#         # Run sims
#         all_traces = {}  # c -> runs x rounds
#         progress = st.progress(0.0)
#         for i, c in enumerate(c_list):
#             traces = []
#             for run in range(n_runs_comp):
#                 trace = run_ucb_trace(c, n_rounds_comp, true_ctrs, return_avg_ctr_over_time=True)
#                 traces.append(trace)
#             all_traces[c] = np.vstack(traces)  # shape (runs, rounds)
#             progress.progress((i + 1) / len(c_list))

#         # Build dataframe for altair
#         records = []
#         for c in all_traces:
#             traces = all_traces[c]  # runs x rounds
#             runs, rounds = traces.shape
#             for run_id in range(runs):
#                 for t in range(rounds):
#                     records.append({"c": float(c), "run": int(run_id + 1), "round": int(t + 1), "avg_ctr": float(traces[run_id, t])})

#         df_long = pd.DataFrame(records)

#         # mean curve per c
#         df_mean = df_long.groupby(["c", "round"], as_index=False)["avg_ctr"].mean()

#         # Plot: mean curves
#         mean_chart = alt.Chart(df_mean).mark_line(point=False).encode(
#             x=alt.X("round:Q", title="Round"),
#             y=alt.Y("avg_ctr:Q", title="Average CTR (cumulative)"),
#             color=alt.Color("c:N", title="c (exploration)")
#         ).properties(height=350, width=900).interactive()

#         # Plot: individual run traces (collapsed)
#         sample_traces = alt.Chart(df_long).mark_line(opacity=0.12).encode(
#             x="round:Q",
#             y="avg_ctr:Q",
#             detail="run:N",
#             color=alt.Color("c:N", title="c")
#         ).properties(height=350, width=900)

#         # Combine
#         placeholder_chart.altair_chart(sample_traces + mean_chart, use_container_width=True)
#         st.success("Comparison runs complete ✅")

#         # show final mean CTR table
#         final_means = df_mean[df_mean["round"] == n_rounds_comp].sort_values("avg_ctr", ascending=False)
#         placeholder_mean.dataframe(final_means.rename(columns={"c": "c", "avg_ctr": f"mean_avg_ctr_at_round_{n_rounds_comp}"}).set_index("c"))

# # -------------------------
# # Tab 4: Statistical selection of best c
# # -------------------------
# with tab4:
#     st.header("4 — Statistical selection: which `c` is best?")
#     st.markdown("Run multiple independent simulations per `c`, collect **average CTR per run**, then use ANOVA + pairwise tests to pick the best `c`.")

#     col1, col2 = st.columns([2, 1])
#     with col2:
#         n_rounds_stat = st.number_input("Rounds per run", min_value=50, max_value=5000, value=500)
#         c_values_text = st.text_input("c values (comma-separated)", "0.1,0.5,1.0,2.0")
#         n_runs_stat = st.number_input("Runs per c", min_value=3, max_value=200, value=30)
#         run_stat = st.button("🔬 Run statistical evaluation")

#     try:
#         c_values_stat = [float(x.strip()) for x in c_values_text.split(",") if x.strip() != ""]
#     except Exception:
#         st.error("Malformed c list")
#         c_values_stat = []

#     with col1:
#         st.write("True CTRs:", [f"{x:.3f}" for x in true_ctrs])
#         stat_placeholder = st.empty()

#     if run_stat and c_values_stat:
#         # run simulations
#         status = st.empty()
#         progress = st.progress(0.0)
#         results = {c: [] for c in c_values_stat}
#         for i, c in enumerate(c_values_stat):
#             status.info(f"Running {n_runs_stat} runs for c={c} ...")
#             for r in range(n_runs_stat):
#                 avg_ctr = run_ucb_trace(c, n_rounds_stat, true_ctrs, return_avg_ctr_over_time=False)
#                 results[c].append(avg_ctr)
#             progress.progress((i + 1) / len(c_values_stat))

#         # Build dataframe
#         df = pd.DataFrame({f"c={c}": results[c] for c in c_values_stat})
#         df.columns = [f"{float(col.split('=')[1]):.6g}" for col in df.columns]  # keep numeric column names (strings)
#         # For plotting convenience convert columns to numeric labels
#         df_plot = pd.DataFrame({float(k): v for k, v in results.items()})

#         # Melt for charts
#         df_melt = df_plot.melt(var_name="c", value_name="avg_ctr")
#         df_melt["c"] = df_melt["c"].astype(float)
#         df_melt = df_melt.sort_values("c")

#         # Boxplot with altair
#         box = alt.Chart(df_melt).mark_boxplot(size=40).encode(
#             x=alt.X("c:Q", title="c (exploration)", sort="ascending"),
#             y=alt.Y("avg_ctr:Q", title="Average CTR"),
#             color=alt.Color("c:Q", legend=None)
#         ).properties(height=350, width=700)

#         st.altair_chart(box, use_container_width=True)

#         # summary stats
#         summary = pd.DataFrame({
#             "c": list(results.keys()),
#             "avg_ctrs_list": list(results.values())
#         })
#         summary["mean"] = summary["avg_ctrs_list"].apply(np.mean)
#         summary["std"] = summary["avg_ctrs_list"].apply(np.std)
#         summary["min"] = summary["avg_ctrs_list"].apply(np.min)
#         summary["max"] = summary["avg_ctrs_list"].apply(np.max)
#         st.subheader("Summary statistics (per c)")
#         st.dataframe(summary[["c", "mean", "std", "min", "max"]].sort_values("mean", ascending=False).assign(mean=lambda d: d["mean"].round(4)))

#         # ANOVA
#         groups = [results[c] for c in c_values_stat]
#         f_stat, p_val = stats.f_oneway(*groups)
#         st.write(f"One-way ANOVA: F = {f_stat:.4f}, p = {p_val:.4e}")
#         if p_val < 0.05:
#             st.success("ANOVA: Significant difference between at least two c values (p < 0.05)")


import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from scipy import stats
import time
import itertools

st.set_page_config(page_title="UCB Tutorial & c-selection", layout="wide")
st.title("🎯 UCB Tutorial — Visualize & Statistically Choose Best `c`")

# -------------------------
# Shared parameters & helpers
# -------------------------
DEFAULT_TRUE_CTRS = [0.05, 0.10, 0.20]

def run_ucb_trace(c, n_rounds, true_ctrs, return_avg_ctr_over_time=False):
    """
    Run a single UCB simulation.
    - If return_avg_ctr_over_time is True, returns array of average CTR (total clicks / rounds) at each round.
    - Otherwise returns avg_ctr (mean clicks per impression) at the end.
    """
    n_ads = len(true_ctrs)
    Q = np.zeros(n_ads)
    N = np.zeros(n_ads)
    clicks = np.zeros(n_rounds, dtype=int)
    cumulative_clicks = 0
    avg_ctr_over_time = np.zeros(n_rounds, dtype=float)

    for t in range(1, n_rounds + 1):
        ucb_values = np.zeros(n_ads)
        for a in range(n_ads):
            if N[a] == 0:
                ucb_values[a] = float("inf")
            else:
                ucb_values[a] = Q[a] + c * np.sqrt(np.log(t + 1) / N[a])

        chosen = int(np.argmax(ucb_values))
        reward = int(np.random.rand() < true_ctrs[chosen])
        clicks[t - 1] = reward
        cumulative_clicks += reward

        # update
        N[chosen] += 1
        Q[chosen] += (reward - Q[chosen]) / N[chosen]
        avg_ctr_over_time[t - 1] = cumulative_clicks / t

    if return_avg_ctr_over_time:
        return avg_ctr_over_time
    else:
        return clicks.mean()  # average CTR over impressions


# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Global settings")
true_ctrs = st.sidebar.text_input("True CTRs (comma-separated)", ",".join(map(str, DEFAULT_TRUE_CTRS)))
try:
    true_ctrs = [float(x.strip()) for x in true_ctrs.split(",") if x.strip() != ""]
    if len(true_ctrs) < 2:
        st.sidebar.error("Provide at least two CTRs (e.g. 0.05,0.10,0.20)")
except Exception:
    st.sidebar.error("Malformed CTRs. Use comma separated floats (e.g. 0.05,0.10,0.20)")
n_rounds_global = st.sidebar.number_input("Default rounds (per run)", min_value=50, max_value=20000, value=500, step=50)
st.sidebar.markdown("---")

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 — UCB primer",
    "2 — Visual single-run",
    "3 — Compare c over time",
    "4 — Statistical selection",
    "5 - Human vs UCB"
])

# -------------------------
# Tab 1: Primer / Definitions
# -------------------------
with tab1:
    st.header("1 — Upper Confidence Bound (UCB) Algorithm Primer")
    
    st.markdown("""
    In **Reinforcement Learning**, an agent interacts with the world and learns consequences of actions through **trial and error**.
    
    One classical problem to formalize **decision-making under uncertainty** is the **Multi-Armed Bandit Problem**:
    - Imagine a slot machine with `k` levers (arms). Each lever gives rewards from an unknown distribution.
    - The agent must choose which lever to pull at each time step to **maximize cumulative reward**.
    
    For example, consider an online advertising trial:
    - An advertiser has 3 different ads (actions).
    - Each user visit corresponds to a round. The advertiser chooses one ad to display.
    - The user may or may not click the ad. The probability of a click is the **true CTR**, which is unknown.
    - The goal: Identify the best ad to maximize clicks while still **learning about all ads**.
    """)

    st.subheader("Action Values and Estimates")
    st.markdown("""
    - **Action-value (true):** $q^*(a)$ — the expected reward for taking action $a$.
    - **Estimated value:** $Q_t(a)$ — the agent's estimate of $q^*(a)$ after $t$ rounds.

    The estimate is updated incrementally using **sample-average method**:
    """)
    st.latex(r"Q_{t+1}(a) = Q_t(a) + \frac{1}{N_t(a)} (R_t - Q_t(a))")
    st.markdown("where $N_t(a)$ is the number of times action $a$ has been selected so far, and $R_t$ is the observed reward at round $t$.")

    st.subheader("Exploration vs Exploitation")
    st.markdown("""
    **Greedy Action (Exploitation):** Choose the action with the highest current estimated value $Q_t(a)$.  
    **Exploration:** Occasionally select other actions to improve knowledge about their rewards.  

    Balancing exploration and exploitation is crucial:
    - Always exploiting may miss the best action.
    - Always exploring wastes potential reward.
    """)

    st.subheader("Upper Confidence Bound (UCB) Action Selection")
    st.markdown("""
    The **UCB algorithm** selects the action $a$ that maximizes:

    """)
    st.latex(r"A_t = \arg\max_a \Big[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \Big]")
    st.markdown("""
    - $Q_t(a)$: Current estimate of action value.  
    - $N_t(a)$: Number of times action $a$ has been selected.  
    - $t$: Current round.  
    - $c$: Exploration parameter controlling how aggressively the algorithm explores.
    """)
    st.latex(r"\sqrt{\frac{\ln t}{N_t(a)}}" )
    st.markdown("""
    - The above term represents uncertainity or exploration
    - Large if $N_t(a)$ is small → action is less explored.
    - Small if $N_t(a)$ is large → action is well-known.
    
    **Principle:** Optimism in the face of uncertainty.  
    Choose actions with the highest **upper confidence bound**, either to gain reward or learn about uncertain actions.
    """)

    st.subheader("Illustrative Example")
    st.markdown("""
    Suppose there are 3 ads with unknown click-through rates.  
    Initially, all actions are explored, UCB selects each ad to reduce uncertainty.  
    Over time, it favors the ads with higher estimated CTR while still occasionally exploring less-chosen ads.
    """)
    
    # Optional: example table to show incremental updates
    example_clicks = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
    Q = 0.0
    table = []
    for i, r in enumerate(example_clicks, start=1):
        N = i
        Q = Q + (r - Q) / N
        table.append({"Impression": i, "R_t (click)": r, "Q_t (estimated CTR)": round(Q, 3)})
    st.subheader("Sample Incremental Updates of Estimated CTR")
    st.table(pd.DataFrame(table))

# -------------------------
# Tab 2: Visual single-run
# -------------------------
# -------------------------
# Tab 2: Visual single-run with step-by-step interactive control
# -------------------------
# -------------------------
# Tab 2: Visual single-run with detailed UCB explanation
# -------------------------
# -------------------------
# Tab 2: Visual single-run with Exploration/Exploitation reasoning
# -------------------------
with tab2:
    st.header("2 — Visual Single Run (Step-by-Step & Interactive)")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("Settings")
        n_rounds = st.number_input("Rounds (impressions)", min_value=10, max_value=1000, value=20)
        c = st.number_input("Exploration parameter c", min_value=0.0, max_value=10.0, value=2.0, step=0.1, format="%.2f")
        st.markdown("**True CTRs (hidden for human):** " + ", ".join([f"{x:.3f}" for x in true_ctrs]))
        st.markdown(
            "We will show **Q-values**, **UCB values**, and reasoning for the ad selection at each round "
            "(whether it was **Exploitation** or **Exploration**)."
        )

    # Initialize session state for interactive simulation
    if "ucb_state" not in st.session_state:
        st.session_state.ucb_state = {
            "round": 0,
            "Q": np.zeros(len(true_ctrs)),
            "N": np.zeros(len(true_ctrs)),
            "clicks": [],
            "ucb_values_history": [],
            "chosen_history": [],
            "Q_history": [],
        }

    state = st.session_state.ucb_state

    # Buttons for controlling steps
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        next_round = st.button("▶️ Next Round")
    with col_btn2:
        prev_round = st.button("◀️ Previous Round")
    with col_btn3:
        reset_state = st.button("🔄 Reset Simulation")

    # Reset simulation
    if reset_state:
        st.session_state.ucb_state = {
            "round": 0,
            "Q": np.zeros(len(true_ctrs)),
            "N": np.zeros(len(true_ctrs)),
            "clicks": [],
            "ucb_values_history": [],
            "chosen_history": [],
            "Q_history": [],
        }
        st.success("Simulation reset. Use 'Next Round' to start again.")
        state = st.session_state.ucb_state

    # Handle previous round
    if prev_round and state["round"] > 0:
        state["round"] -= 1
        state["clicks"].pop()
        state["ucb_values_history"].pop()
        state["chosen_history"].pop()
        state["Q_history"].pop()
        state["Q"] = np.zeros(len(true_ctrs))
        state["N"] = np.zeros(len(true_ctrs))
        # Recompute Q, N based on remaining history
        for r_idx, chosen in enumerate(state["chosen_history"]):
            reward = state["clicks"][r_idx]
            state["N"][chosen] += 1
            state["Q"][chosen] += (reward - state["Q"][chosen]) / state["N"][chosen]

    # Handle next round
    if next_round and state["round"] < n_rounds:
        state["round"] += 1
        t = state["round"]

        # Save previous Q for explanation
        prev_Q = state["Q"].copy()

        # Compute UCB values
        ucb_values = np.zeros(len(true_ctrs))
        for a in range(len(true_ctrs)):
            if state["N"][a] == 0:
                ucb_values[a] = float("inf")
            else:
                ucb_values[a] = state["Q"][a] + c * np.sqrt(np.log(t + 1) / state["N"][a])

        # Select ad
        chosen = int(np.argmax(ucb_values))
        reward = int(np.random.rand() < true_ctrs[chosen])

        # Update estimates
        state["N"][chosen] += 1
        state["Q"][chosen] += (reward - state["Q"][chosen]) / state["N"][chosen]

        # Save histories
        state["ucb_values_history"].append(ucb_values)
        state["chosen_history"].append(chosen)
        state["clicks"].append(reward)
        state["Q_history"].append(prev_Q)

    # -------------------------
    # Display
    # -------------------------
    st.subheader(f"Round {state['round']}/{n_rounds}")

    # Show ad visuals
    ad_cols = st.columns(len(true_ctrs))
    for i, col in enumerate(ad_cols):
        if state["round"] > 0 and i == state["chosen_history"][-1]:
            col.markdown(
                f"<div style='border:3px solid #2ecc71;padding:10px;border-radius:8px;text-align:center;background:#f0fff5;'>"
                f"<h4>🟩 Ad {i+1}</h4><p>Displayed — {'✅ Click' if state['clicks'][-1] else '❌ No click'}</p></div>",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f"<div style='border:1px solid #ddd;padding:10px;border-radius:8px;text-align:center;background:#fafafa;'>"
                f"<h4>⬜ Ad {i+1}</h4><p>Idle</p></div>",
                unsafe_allow_html=True,
            )

    # -------------------------
    # Explanation table
    # -------------------------
    if state["round"] > 0:
        r_idx = state["round"] - 1
        explanation = []
        prev_Q = state["Q_history"][r_idx]
        ucb_vals = state["ucb_values_history"][r_idx]
        chosen = state["chosen_history"][r_idx]
        reward = state["clicks"][r_idx]

        # Determine reason: Exploration or Exploitation
        max_Q = np.max(prev_Q)
        if prev_Q[chosen] == max_Q:
            reason_text = "Exploitation (selected ad had highest Q)"
        else:
            reason_text = "Exploration (selected ad did NOT have highest Q)"

        for i in range(len(true_ctrs)):
            explanation.append({
                "Ad": i+1,
                "Previous Q (Estimated CTR)": round(prev_Q[i],3),
                "N shown": int(state["N"][i]),
                "UCB Value": round(ucb_vals[i],3),
                "Reason for selection": reason_text if i == chosen else ""
            })

        st.subheader("Detailed UCB Explanation for this Round")
        st.dataframe(pd.DataFrame(explanation))
        st.markdown(
            "**Formula used:**  UCB(a) = Q(a) + c * sqrt( log(t) / N(a) )\n\n"
            "- Q(a) = current estimated CTR for ad a\n"
            "- c = exploration parameter\n"
            "- t = current round\n"
            "- N(a) = times ad a has been selected\n\n"
            "The ad with the **highest UCB value** is selected, which balances **exploration** and **exploitation**. "
            "You can increase `c` to see more explorations, or decrease it to favor exploitation."
        )


# -------------------------
# Tab 3: Compare c over time (fixed)
# -------------------------
with tab3:
    st.header("3 — Compare different `c` values over time (fixed)")
    st.markdown("Run several independent simulations per `c`, then plot average CTR vs rounds (learning curve) and per-run traces.")

    colL, colR = st.columns([2, 1])
    with colR:
        n_rounds_comp = st.number_input("Rounds for comparison", min_value=50, max_value=5000, value=500, step=50, key="n_rounds_comp")
        c_list_input = st.text_input("c values (comma-separated)", "0.1,0.5,1.0,2.0,4.0", key="c_list_input_tab3")
        n_runs_comp = st.number_input("Runs per c (for averaging)", min_value=1, max_value=50, value=8, key="n_runs_comp")
        max_plot_runs = st.slider("Max number of per-c run traces to plot (for performance)", 1, min(20, n_runs_comp), min(8, n_runs_comp))
        run_comp = st.button("📈 Run comparison (Tab 3)")

    # parse c list safely
    try:
        c_list = sorted([float(x.strip()) for x in c_list_input.split(",") if x.strip() != ""])
    except Exception:
        st.error("Malformed c list")
        c_list = []

    with colL:
        placeholder_chart = st.empty()
        placeholder_mean = st.empty()

    if run_comp and c_list:
        progress = st.progress(0.0)
        all_traces = {}  # c -> runs x rounds

        # run sims
        for i, c in enumerate(c_list):
            traces = []
            for run in range(n_runs_comp):
                trace = run_ucb_trace(c, n_rounds_comp, true_ctrs, return_avg_ctr_over_time=True)
                traces.append(trace)
            all_traces[c] = np.vstack(traces)  # shape (runs, rounds)
            progress.progress((i + 1) / len(c_list))

        # Build long DataFrame for Altair
        records = []
        for c in sorted(all_traces.keys()):
            traces = all_traces[c]  # runs x rounds
            runs, rounds = traces.shape
            for run_id in range(runs):
                for t in range(rounds):
                    records.append({
                        "c": float(c),
                        "run": int(run_id + 1),
                        "round": int(t + 1),
                        "avg_ctr": float(traces[run_id, t])
                    })
        df_long = pd.DataFrame.from_records(records)

        # ensure ordering and types
        df_long["c"] = df_long["c"].astype(float)
        df_long = df_long.sort_values(["c", "run", "round"]).reset_index(drop=True)

        # create a unique run identifier per (c, run) so Altair treats each trace separately
        df_long["run_uid"] = df_long["c"].astype(str) + "_run" + df_long["run"].astype(str)

        # compute mean curve per (c, round)
        df_mean = df_long.groupby(["c", "round"], as_index=False)["avg_ctr"].mean()

        # Subsample per-c runs for plotting if requested (avoid plotting too many lines)
        df_long_plot = df_long.groupby("c").apply(
            lambda g: g[g["run"].isin(sorted(g["run"].unique())[:max_plot_runs])]
        ).reset_index(drop=True)

        # Altair charts
        # faint individual run traces (colored by c)
        individual_traces = (
            alt.Chart(df_long_plot)
            .mark_line(opacity=0.12)
            .encode(
                x=alt.X("round:Q", title="Round"),
                y=alt.Y("avg_ctr:Q", title="Average CTR (cumulative)"),
                color=alt.Color("c:N", title="c (exploration)", sort=sorted(map(str, c_list))),
                detail="run_uid:N"
            )
            .properties(height=360)
        )

        # bold mean curves per c
        mean_curves = (
            alt.Chart(df_mean)
            .mark_line(point=False, strokeWidth=3)
            .encode(
                x="round:Q",
                y="avg_ctr:Q",
                color=alt.Color("c:N", title="c (exploration)", sort=sorted(map(str, c_list)))
            )
        )

        # overlay and display
        layered = individual_traces + mean_curves
        placeholder_chart.altair_chart(layered.interactive(), use_container_width=True)

        # show final mean CTRs at last round
        final_means = df_mean[df_mean["round"] == n_rounds_comp].sort_values("avg_ctr", ascending=False)
        placeholder_mean.subheader("Final mean CTRs at last round")
        placeholder_mean.dataframe(final_means.rename(columns={"avg_ctr": f"mean_avg_ctr_at_round_{n_rounds_comp}"}).set_index("c"))

        st.success("Comparison runs complete ✅")

# -------------------------
# Tab 4: Statistical selection of best c
# -------------------------
# -------------------------
# Tab 4: Statistical selection of best c (enhanced)
# -------------------------
with tab4:
    st.header("4 — Statistical selection: which `c` is best?")
    st.markdown("Run multiple simulations per `c`, compute average CTRs, perform ANOVA + pairwise t-tests, and visualize results.")

    col1, col2 = st.columns([2, 1])
    with col2:
        n_rounds_stat = st.number_input("Rounds per run", min_value=50, max_value=5000, value=500)
        c_values_text = st.text_input("c values (comma-separated)", "0.1,0.5,1.0,2.0,4.0")
        n_runs_stat = st.number_input("Runs per c", min_value=3, max_value=200, value=30)
        run_stat = st.button("🔬 Run statistical evaluation")

    try:
        c_values_stat = [float(x.strip()) for x in c_values_text.split(",") if x.strip() != ""]
    except Exception:
        st.error("Malformed c list")
        c_values_stat = []

    with col1:
        st.write("True CTRs:", [f"{x:.3f}" for x in true_ctrs])

    if run_stat and c_values_stat:
        # Run simulations
        progress = st.progress(0.0)
        results = {c: [] for c in c_values_stat}

        for i, c in enumerate(c_values_stat):
            for r in range(n_runs_stat):
                avg_ctr = run_ucb_trace(c, n_rounds_stat, true_ctrs, return_avg_ctr_over_time=False)
                results[c].append(avg_ctr)
            progress.progress((i + 1) / len(c_values_stat))

        # --- Build dataframe
        df_plot = pd.DataFrame({float(k): v for k, v in results.items()})
        df_melt = df_plot.melt(var_name="c", value_name="avg_ctr").sort_values("c")

        # --- Boxplot of reward (CTR) distribution per c
        st.subheader("📦 Reward (CTR) Distribution per c Value")
        box = alt.Chart(df_melt).mark_boxplot(size=50).encode(
            x=alt.X("c:Q", title="c value"),
            y=alt.Y("avg_ctr:Q", title="Average CTR per Run"),
            color=alt.Color("c:Q", legend=None)
        ).properties(height=350, width=700)
        st.altair_chart(box, use_container_width=True)

        # --- Average CTR per c (line plot)
        summary_df = df_melt.groupby("c", as_index=False).agg(avg_ctr=("avg_ctr", "mean"))
        st.subheader("📈 Average CTR across Runs vs c Value")
        base = alt.Chart(summary_df).mark_line(point=True).encode(
            x=alt.X("c:Q", title="c Value"),
            y=alt.Y("avg_ctr:Q", title="Mean CTR"),
            tooltip=["c", "avg_ctr"]
        ).properties(height=300, width=700)
        st.altair_chart(base, use_container_width=True)

        # --- CTR table across runs
        st.subheader("📊 CTR Values Across Different Runs")
        st.dataframe(df_plot.style.highlight_max(axis=0))

        # --- Summary statistics
        st.subheader("Summary Statistics")
        summary_stats = df_melt.groupby("c")["avg_ctr"].agg(["mean", "std", "min", "max"]).reset_index()
        st.dataframe(summary_stats.style.format({"mean": "{:.4f}", "std": "{:.4f}", "min": "{:.4f}", "max": "{:.4f}"}))

        # --- ANOVA
        groups = [results[c] for c in c_values_stat]
        f_stat, p_val = stats.f_oneway(*groups)
        st.write(f"**One-way ANOVA:** F = {f_stat:.4f}, p = {p_val:.4e}")

        # --- Pairwise t-tests
        pairs = list(itertools.combinations(c_values_stat, 2))
        pvals = []
        for (c1, c2) in pairs:
            _, p = stats.ttest_ind(results[c1], results[c2], equal_var=False)
            pvals.append({"c1": c1, "c2": c2, "p_value": p, "significant": p < 0.05})
        df_pairs = pd.DataFrame(pvals)

        st.subheader("Pairwise T-Test Results")
        st.dataframe(df_pairs)

        # --- Determine best c value
        if not df_pairs.empty:
            df_sig = df_pairs[df_pairs["significant"]]
            if not df_sig.empty:
                sig_best = pd.concat([df_sig["c1"], df_sig["c2"]]).unique()
                best_row = summary_df.loc[summary_df["avg_ctr"].idxmax()]
                best_c = best_row["c"]
                best_ctr = best_row["avg_ctr"]
                min_p = df_sig["p_value"].min()

                st.success(
                    f"""
                    ✅ **Best performing c value:** {best_c:.2f}  
                    📈 **Highest average CTR:** {best_ctr:.4f}  
                    🧠 **Smallest significant p-value:** {min_p:.4e}  
                    🔍 **Statistically different (significant) c values:** {sig_best.tolist()}
                    """
                )
            else:
                st.info("No statistically significant differences found between c values.")
        else:
            st.info("No pairwise comparisons available for statistical testing.")


# -------------------------
# Tab 5: Human vs UCB game
# -------------------------

# tab5 = st.tab("5 — Human vs UCB Game")

# -------------------------
# Tab 5: Human vs UCB game
# -------------------------
with tab5:
    st.header("5 — Human vs UCB Game Simulation")
    st.markdown("**Game Rules:**")
    st.markdown("""
    1. There are 3 ads with hidden CTRs.  
    2. In each round, you select one ad to display.  
    3. A reward (click/no click) is randomly generated based on the true CTR.  
    4. UCB simultaneously selects an ad based on its current estimates.  
    5. After each round, a table shows the user and UCB choices, rewards, and estimated CTRs.  
    6. The game ends when you press **End Game**. The winner is revealed, and actual CTRs are shown.
    """)

    n_rounds_game = st.number_input("Total rounds (max)", min_value=10, max_value=500, value=20, step=1)
    c_game = st.number_input("Exploration parameter c for UCB", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

    # Initialize game state if not present
    if "game_state" not in st.session_state:
        st.session_state.game_state = {
            "round": 0,
            "user_clicks": 0,
            "ucb_clicks": 0,
            "user_total": [],
            "ucb_total": [],
            "Q": np.zeros(len(true_ctrs)),
            "N": np.zeros(len(true_ctrs)),
            "human_Q": np.zeros(len(true_ctrs)),
            "human_N": np.zeros(len(true_ctrs)),
            "table": pd.DataFrame(columns=[
                "Round", "User Choice", "User Reward", "Estimated CTR Human",
                "UCB Choice", "UCB Reward", "Estimated CTR UCB"
            ])
        }

    game_state = st.session_state.game_state
    human_Q = game_state["human_Q"]
    human_N = game_state["human_N"]

    col_user, col_actions = st.columns([1, 2])
    with col_user:
        st.subheader("Select your ad")
        user_choice = st.radio("Pick an ad:", options=[f"Ad {i+1}" for i in range(len(true_ctrs))])
        user_choice_idx = int(user_choice.split()[-1]) - 1

    with col_actions:
        end_game = st.button("🏁 End Game / Reveal Results")
        next_round = st.button("▶️ Play Next Round")

    # --- Play next round ---
    if next_round and game_state["round"] < n_rounds_game:
        game_state["round"] += 1
        t = game_state["round"]

        # Human reward & update human estimates
        user_reward = int(np.random.rand() < true_ctrs[user_choice_idx])
        game_state["user_clicks"] += user_reward
        game_state["user_total"].append(user_reward)

        human_N[user_choice_idx] += 1
        human_Q[user_choice_idx] += (user_reward - human_Q[user_choice_idx]) / human_N[user_choice_idx]

        # UCB selection & reward
        Q, N = game_state["Q"], game_state["N"]
        ucb_values = np.zeros(len(true_ctrs))
        for a in range(len(true_ctrs)):
            if N[a] == 0:
                ucb_values[a] = float("inf")
            else:
                ucb_values[a] = Q[a] + c_game * np.sqrt(np.log(t + 1) / N[a])
        ucb_choice = int(np.argmax(ucb_values))
        ucb_reward = int(np.random.rand() < true_ctrs[ucb_choice])
        game_state["ucb_clicks"] += ucb_reward
        game_state["ucb_total"].append(ucb_reward)

        # Update UCB estimates
        N[ucb_choice] += 1
        Q[ucb_choice] += (ucb_reward - Q[ucb_choice]) / N[ucb_choice]

        # Append round info to table
        game_state["table"] = pd.concat([
            game_state["table"],
            pd.DataFrame([{
                "Round": t,
                "User Choice": f"Ad {user_choice_idx+1}",
                "User Reward": user_reward,
                "Estimated CTR Human": np.round(human_Q, 3).tolist(),
                "UCB Choice": f"Ad {ucb_choice+1}",
                "UCB Reward": ucb_reward,
                "Estimated CTR UCB": np.round(Q, 3).tolist()
            }])
        ], ignore_index=True)

    # --- Display game log ---
    st.subheader("Game Log")
    st.dataframe(game_state["table"])

    # --- End game results ---
    if end_game or game_state["round"] >= n_rounds_game:
        st.subheader("🏆 Game Results")
        st.markdown(f"**Your total clicks:** {game_state['user_clicks']}  |  **UCB total clicks:** {game_state['ucb_clicks']}")
        if game_state["user_clicks"] > game_state["ucb_clicks"]:
            st.success("🎉 You win! Human intuition beats UCB.")
        elif game_state["user_clicks"] < game_state["ucb_clicks"]:
            st.success("🤖 UCB wins! Machine learning outperforms human intuition.")
        else:
            st.info("🤝 It's a tie!")

        st.markdown(f"**Actual CTRs of the ads:** {true_ctrs}")
        st.markdown("Game finished. You can reset the page to play again.")
