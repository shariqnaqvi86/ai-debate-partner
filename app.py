
"""
AI Debate Partner & Coach — Georgetown Prototype
"""

import uuid
import datetime
import os
from dotenv import load_dotenv

import streamlit as st
import plotly.graph_objects as go

from src.debate import (
    build_debate_prompt,
    infer_debate_topic,
    is_explicit_topic_switch_turn,
    is_substantive_debate_turn,
    is_valid_topic_label,
    topic_turn_strength,
)
from src.evidence import is_evidence_request
from src.llm_client import LLMClient, MockLLMClient
from src.scoring import score_turn_with_llm, ScoringOutput
from src.source_lookup import lookup_relevant_source_hits
from src.storage import append_log, export_session, export_session_markdown
from src.subagent import spin_off_research_subagent, get_synthesized_counter_evidence
load_dotenv(override=False)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

PERSONAS = {
    "Public Health Official (Harm Reduction)": {
        "description": (
            "A seasoned public health leader with 20+ years in community medicine, "
            "formerly a frontline clinician in underserved urban communities. Now directs "
            "a city health department focused on evidence-based, equity-centered policy."
        ),
        "core_beliefs": [
            "Abstinence-only approaches have repeatedly failed — meeting people where they are saves lives.",
            "Health outcomes are inseparable from race, income, housing, and systemic inequity.",
            "Person-first language ('person who uses drugs') reduces stigma and improves care engagement.",
            "Government's role is to reduce suffering, not to moralize behavior.",
            "Incremental, pragmatic progress beats ideological purity.",
        ],
        "debate_style": (
            "Leads with epidemiological data and peer-reviewed studies. Humanizes statistics "
            "with real patient stories. Stays calm under pressure — frames opposition arguments "
            "as well-intentioned but empirically flawed. Frequently references WHO, CDC, and "
            "Lancet findings. Pushes back on moral framing by redirecting to measurable outcomes."
        ),
        "rhetorical_tendencies": [
            "Uses phrases like 'the evidence consistently shows...', 'when we look at the data...'",
            "Reframes moral arguments as public health questions: 'The question isn't right or wrong — it's what works.'",
            "Acknowledges the other side's concern before dismantling it: 'I understand the instinct to...'",
            "Invokes marginalized communities as the true test of any policy.",
        ],
        "common_arguments": [
            "Needle exchanges reduce HIV transmission without increasing drug use (Vancouver, Portugal data).",
            "Criminalization drives people away from treatment — decriminalization improves outcomes.",
            "Naloxone access saves lives; moral objections cost lives.",
            "Safe injection sites reduce overdose deaths and connect people to care.",
            "Abstinence-only sex ed is correlated with higher teen pregnancy rates.",
        ],
        "vulnerabilities": [
            "Can appear to minimize personal accountability or community values.",
            "Critics argue harm reduction 'enables' dangerous behavior.",
            "Difficult to defend politically when high-profile incidents occur near harm-reduction sites.",
        ],
        "tone": "Empathetic, clinical, quietly passionate. Measured but firm.",
    },

    "Public Health Official (Abstinence / Zero-Tolerance)": {
        "description": (
            "A senior public health administrator with a background in behavioral psychology "
            "and faith-community outreach. Has led state-level prevention campaigns and advises "
            "legislative committees on drug, alcohol, and sexual health policy."
        ),
        "core_beliefs": [
            "Prevention is always superior to mitigation — we should stop harm before it starts.",
            "Clear, unambiguous standards send a social message that shapes behavior over time.",
            "Harm reduction programs can normalize and inadvertently encourage risky behavior.",
            "Community values, family structure, and personal responsibility are protective factors.",
            "The state should set high standards, not accommodate the lowest common denominator.",
        ],
        "debate_style": (
            "Leads with prevention data, long-term population-level trends, and moral-community "
            "framing. Appeals to parental rights, cultural norms, and the social contract. "
            "Challenges harm-reduction evidence as short-term and narrow. Questions whether "
            "society should 'subsidize' destructive choices. Frequently invokes fiscal "
            "responsibility and downstream societal costs, and references federal guidance "
            "that limits SAMHSA funding to overdose reversal, prevention, and recovery services while "
            "forbid[s] support for syringes, smoking kits, test strips, or drug-use paraphernalia."
        ),
        "rhetorical_tendencies": [
            "Uses phrases like 'we have to ask what message this sends...', 'where do we draw the line?'",
            "Reframes data debates as values debates: 'Even if it reduces X, is this the society we want?'",
            "Appeals to majority sentiment and democratic legitimacy.",
            "Raises slippery-slope concerns about normalizing dangerous behaviors.",
        ],
        "common_arguments": [
            "Abstinence, when actually practiced, is 100% effective — the goal should be behavior change.",
            "Harm reduction shifts responsibility from the individual to the taxpayer.",
            "Zero-tolerance drug policies in schools create safer learning environments.",
            "Countries with stricter drug enforcement (Japan, Singapore) have very low addiction rates.",
            "Federal SAMHSA guidance now favors funding overdose reversal, wound care, infectious disease prevention, and recovery referrals while barring support for syringes, pipes, fentanyl test strips, and other paraphernalia that facilitate illicit drug use.",
            "Providing 'safe' means for unsafe behavior undercuts the deterrent effect of consequences.",
        ],
        "vulnerabilities": [
            "Abstinence-only programs have poor empirical track records in many contexts.",
            "Zero-tolerance can produce unjust outcomes (e.g., mandatory minimums, racial disparities).",
            "Difficult to defend when enforcement disproportionately impacts minority communities.",
            "Critics argue this approach prioritizes ideology over lives lost.",
        ],
        "tone": "Authoritative, principled, occasionally impassioned. Projects moral clarity.",
    },

    "State Legislator (Harm Reduction)": {
        "description": (
            "A two-term state representative from a mid-sized urban district, formerly a social worker "
            "and public defender. Serves on the Health & Human Services Committee and is known for "
            "bipartisan coalition-building around practical harm-reduction policy."
        ),
        "core_beliefs": [
            "Government must protect constituents through policies that measurably reduce harm.",
            "Criminalization is costly and often less effective than treatment and diversion.",
            "Legislative persuasion requires fiscal realism, public safety framing, and implementation detail.",
            "Evidence-based policy is essential in hearings and budget negotiations.",
            "Cross-sector partnerships with law enforcement and health agencies are pragmatic, not ideological.",
        ],
        "debate_style": (
            "Frames arguments around budget impact, recidivism, workforce outcomes, and constituent safety. "
            "Uses state-policy comparisons and implementation design details, and emphasizes coalition language "
            "that can survive committee scrutiny and floor votes."
        ),
        "rhetorical_tendencies": [
            "Starts with district/state-level framing and practical consequences.",
            "Uses fiscal reframing: prevention/treatment costs versus incarceration and emergency burden.",
            "Deflects moral binaries toward measurable taxpayer outcomes.",
            "References comparable-state policy outcomes to strengthen feasibility claims.",
        ],
        "common_arguments": [
            "Treatment/diversion is often less costly than incarceration for similar populations.",
            "Good Samaritan and harm-reduction policies can reduce mortality without increasing use.",
            "Medicaid and access policy design strongly affect treatment uptake.",
            "Drug-court and diversion design can improve recidivism outcomes versus pure punishment.",
            "Legislative design quality determines whether evidence-based programs scale effectively.",
        ],
        "vulnerabilities": [
            "Can be attacked as 'soft on crime' in polarized environments.",
            "Legislative implementation varies sharply by county capacity.",
            "Funding constraints and procurement timelines can blunt early outcomes.",
        ],
        "preferred_sources": [
            "NCSL",
            "Pew Charitable Trusts",
            "Vera Institute",
            "KFF",
            "SAMHSA",
            "CDC",
            "Stateline (Pew)",
            "National Drug Court Institute",
        ],
        "tone": "Pragmatic, constituent-focused, bipartisan, and implementation-oriented.",
    },

    "State Legislator (Abstinence / Zero-Tolerance)": {
        "description": (
            "A three-term state senator from a rural district with strong law-enforcement and faith-community ties. "
            "Former county sheriff and current chair of Judiciary and Public Safety, focused on deterrence and local control."
        ),
        "core_beliefs": [
            "Law should reinforce community standards and deterrence.",
            "Permissive policy can normalize harmful behavior and increase downstream burden.",
            "Local communities should retain policy control where possible.",
            "Personal responsibility and family stability are key protective factors.",
            "Visible enforcement can function as prevention.",
        ],
        "debate_style": (
            "Leads with public safety, rule-of-law framing, and local governance concerns. "
            "Challenges external generalizability of studies, questions unfunded mandates, and emphasizes "
            "social-signaling effects of policy choices."
        ),
        "rhetorical_tendencies": [
            "Uses message-signaling language around youth and community norms.",
            "Presses external validity concerns for out-of-state or urban-only findings.",
            "Raises slippery-slope and implementation-enforcement burden concerns.",
            "Challenges narrow fiscal math by adding broader social-cost framing.",
        ],
        "common_arguments": [
            "Strict enforcement and clear standards can support deterrence in some contexts.",
            "Policy design must account for local capacity and public buy-in.",
            "School-zone and youth-protection statutes remain politically durable.",
            "Prevention-first frameworks can be framed as long-horizon cost control.",
            "Community opposition can indicate implementation legitimacy risks.",
        ],
        "vulnerabilities": [
            "Pure enforcement models face criticism on cost and equity grounds.",
            "Abstinence-only evidence is mixed across many peer-reviewed reviews.",
            "Opioid burden in rural communities can weaken punitive-only messaging.",
        ],
        "preferred_sources": [
            "NCSL",
            "NGA",
            "DEA",
            "ONDCP",
            "Heritage Foundation",
            "SAMHSA (prevention data only)",
            "Bureau of Justice Statistics",
            "Manhattan Institute",
        ],
        "tone": "Authoritative, values-driven, and community-anchored.",
    },
}

ROLE_OPTIONS = [
    "Public Health Official",
    "State Legislator",
    "Law Enforcement",
    "Clinician",
    "Community Advocate",
    "Federal Drug Policy Czar",
    "City Mayor / Local Official",
    "Policy Analyst / Bioethicist",
    "Custom Persona",
]
ORIENTATION_OPTIONS = [
    "Minimal Harm Reduction",
    "Harm Reduction",
    "Abstinence / Zero-Tolerance",
]

ROLE_DESCRIPTIONS = {
    "Public Health Official": "A population-health leader who centers evidence, health equity, and practical public-health outcomes.",
    "State Legislator": "A policymaker focused on budget, feasibility, legal authority, and building support across constituencies.",
    "Law Enforcement": "A public safety leader concerned with enforcement, order, community trust, and operational impacts.",
    "Clinician": "A healthcare provider who prioritizes patient safety, treatment access, and real-world clinical outcomes.",
    "Community Advocate": "A grassroots voice that centers lived experience, stigma reduction, and neighborhood-level impact.",
    "Federal Drug Policy Czar": "A high-level federal executive managing national drug strategy, interagency coordination, and federal funding.",
    "City Mayor / Local Official": "A municipal leader balancing public safety, economic vitality, local business concerns, and public health.",
    "Policy Analyst / Bioethicist": "An academic researcher evaluating policy ethics, systemic tradeoffs, cost-benefit models, and evidence validity.",
    "Custom Persona": "Define your own custom AI debate stakeholder role.",
}

ORIENTATION_DESCRIPTIONS = {
    "Minimal Harm Reduction": "A cautious, evidence-informed approach that prioritizes immediate overdose reversal, infection prevention, and low-barrier services while avoiding expansive policy commitments.",
    "Harm Reduction": "A pragmatic public-health approach that seeks to reduce harm, preserve life, and connect people to care through a broader set of evidence-based interventions.",
    "Abstinence / Zero-Tolerance": "A prevention-first stance that emphasizes clear behavioral standards, structured support, and enforcement to discourage substance use.",
}


def build_persona_label(role: str, orientation: str) -> str:
    return f"{role} ({orientation})"


DEBATE_MODES = [
    "Adversary Mode",
    "Socratic Mode",
    "Steel-Manning Mode",
]

DEBATE_MODE_DESCRIPTIONS = {
    "Adversary Mode": "Presents strong counter-arguments, attacks logical flaws, and pushes back aggressively in character.",
    "Socratic Mode": "Asks probing policy questions, examines underlying assumptions, and tests internal consistency.",
    "Steel-Manning Mode": "First articulates and strengthens the best version of your argument, then presents a surgical refutation.",
}


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "session_id": str(uuid.uuid4()),
        "transcript": [],
        "scores": {"ethos": 0, "logos": 0, "pathos": 0},
        "rewrites": [],
        "reflections": [],
        "flags": [],
        "evidence_items": [],
        "role": ROLE_OPTIONS[0],
        "orientation": ORIENTATION_OPTIONS[0],
        "persona": build_persona_label(ROLE_OPTIONS[0], ORIENTATION_OPTIONS[0]),
        "debate_mode": DEBATE_MODES[0],
        "practice_mode": "Open practice",
        "scoring_rationale": {},
        "debate_topic": "",
        "topic_source": "pending",
        "topic_strength": 0,
        "reply_error": "",
        "scoring_error": "",
        "scoring_note": "Scores will appear after your first substantive argument.",
        "input_locked": False,
        "pending_user_input": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Debate Partner & Coach",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.9rem; padding-bottom: 1rem; }
    button[title="View fullscreen"] { display: none !important; }
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem !important; }
    .stChatMessage { border-radius: 10px; }
    h3 { white-space: normal !important; overflow: visible !important; text-overflow: unset !important; }
    .flag-box { background:#fff3cd; border-left:4px solid #ffc107; padding:8px 12px; border-radius:4px; margin-bottom:6px; }
    .rewrite-box { background:#d1ecf1; border-left:4px solid #17a2b8; padding:8px 12px; border-radius:4px; margin-bottom:6px; }
    .reflect-box { background:#d4edda; border-left:4px solid #28a745; padding:8px 12px; border-radius:4px; margin-bottom:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — SETTINGS & EVIDENCE
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/d/d9/Georgetown_University_seal.svg", width=60)
    st.title("Debate Coach")
    st.caption("Georgetown University — Policy Debate Prototype")
    st.divider()

    st.subheader("⚙️ Session Setup")
    role_col, orientation_col = st.columns([1, 1])
    new_role = role_col.selectbox(
        "AI Role",
        ROLE_OPTIONS,
        index=ROLE_OPTIONS.index(st.session_state["role"]),
        key="role_select",
    )
    new_orientation = orientation_col.selectbox(
        "Policy orientation",
        ORIENTATION_OPTIONS,
        index=ORIENTATION_OPTIONS.index(st.session_state["orientation"]),
        key="orientation_select",
    )
    st.markdown(
        "<div style='border:1px solid #dfe3e8; border-radius:12px; padding:16px; background:#f7fbff; margin-top:10px;'>"
        f"<strong style='font-size:1.05em;'>Selected persona:</strong> {build_persona_label(new_role, new_orientation)}<br><br>"
        f"<strong>Role</strong>: {new_role}<br>"
        f"<span style='color:#2c3e50; font-size:0.95em;'>{ROLE_DESCRIPTIONS.get(new_role, '')}</span><br><br>"
        f"<strong>Policy orientation</strong>: {new_orientation}<br>"
        f"<span style='color:#2c3e50; font-size:0.95em;'>{ORIENTATION_DESCRIPTIONS.get(new_orientation, '')}</span><br><br>"
        "<em style='color:#5a6370; font-size:0.92em;'>Role defines the professional perspective of the AI speaker. Orientation defines the policy stance the AI will adopt.</em>"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Orientation spectrum", expanded=False):
        st.markdown(
            "- **Minimal Harm Reduction:** a narrowly scoped policy stance focused on immediate life-saving interventions and pragmatic prevention while avoiding broader system changes.  \n"
            "- **Harm Reduction:** a comprehensive public-health approach that prioritizes continuity of care, risk mitigation, and evidence-based interventions to reduce morbidity and mortality.  \n"
            "- **Abstinence / Zero-Tolerance:** a prevention-first posture that emphasizes clear behavioral expectations, enforcement, and messaging aimed at discouraging substance use."
        )
    new_debate_mode = st.selectbox(
        "Debate Mode",
        DEBATE_MODES,
        index=DEBATE_MODES.index(st.session_state.get("debate_mode", DEBATE_MODES[0])),
        key="debate_mode_select",
    )
    st.caption(f"_{DEBATE_MODE_DESCRIPTIONS.get(new_debate_mode, '')}_")

    practice_modes = [
        "Open practice",
        "Affirmative claim practice",
        "Opposition / rebuttal practice",
        "Evidence sourcing practice",
    ]
    new_practice_mode = st.selectbox("Practice mode", practice_modes, key="practice_mode_select")

    if new_debate_mode != st.session_state.get("debate_mode"):
        st.session_state["debate_mode"] = new_debate_mode
        st.session_state["scoring_note"] = f"Switched to {new_debate_mode}."

    if new_role != st.session_state["role"] or new_orientation != st.session_state["orientation"]:
        st.session_state["role"] = new_role
        st.session_state["orientation"] = new_orientation
        st.session_state["persona"] = build_persona_label(new_role, new_orientation)
        st.session_state["transcript"] = []
        st.session_state["scores"] = {"ethos": 0, "logos": 0, "pathos": 0}
        st.session_state["scoring_rationale"] = {}
        st.session_state["rewrites"] = []
        st.session_state["reflections"] = []
        st.session_state["flags"] = []
        st.session_state["debate_topic"] = ""
        st.session_state["topic_source"] = "pending"
        st.session_state["topic_strength"] = 0
        st.session_state["reply_error"] = ""
        st.session_state["scoring_error"] = ""
        st.session_state["scoring_note"] = "Scores will appear after your first substantive argument."

    if new_practice_mode != st.session_state["practice_mode"]:
        st.session_state["practice_mode"] = new_practice_mode
        st.session_state["scoring_note"] = "Practice mode changed. Continue with the same conversation or adjust your approach."

    st.subheader("🧭 Debate Topic")
    if st.session_state["debate_topic"]:
        with st.form("topic_editor_form"):
            edited_topic = st.text_area(
                "Detected topic",
                value=st.session_state["debate_topic"],
                height=90,
            )
            save_topic = st.form_submit_button("Update Topic")
            if save_topic and edited_topic.strip():
                st.session_state["debate_topic"] = edited_topic.strip()
                st.session_state["topic_source"] = "manual"
                st.session_state["topic_strength"] = 2
                st.rerun()
        if st.session_state["topic_source"] == "manual":
            st.caption("Topic manually refined for this session.")
        else:
            st.caption("Topic inferred from the conversation. You can refine the wording here if needed.")
    else:
        st.info("The app will infer the debate topic from the first substantive message and keep it for the session.")

    st.divider()
    st.subheader("📚 Evidence Panel")
    st.caption("Add up to 3 evidence items (PubTrawlr / LegiScan)")

    ev_count = len(st.session_state["evidence_items"])
    if ev_count < 3:
        with st.form("add_evidence_form", clear_on_submit=True):
            ev_title = st.text_input("Source title")
            ev_url = st.text_input("URL")
            ev_b1 = st.text_input("Takeaway 1")
            ev_b2 = st.text_input("Takeaway 2")
            submitted = st.form_submit_button("➕ Add Evidence")
            if submitted and ev_title:
                st.session_state["evidence_items"].append({
                    "title": ev_title,
                    "url": ev_url,
                    "bullets": [ev_b1, ev_b2],
                })
                st.rerun()

    for i, ev in enumerate(st.session_state["evidence_items"]):
        with st.expander(f"📄 {ev['title'][:40]}", expanded=False):
            st.markdown(f"[🔗 Link]({ev['url']})" if ev["url"] else "_No URL_")
            for b in ev["bullets"]:
                if b:
                    st.markdown(f"• {b}")
            if st.button("🗑 Remove", key=f"rm_{i}"):
                st.session_state["evidence_items"].pop(i)
                st.rerun()

    st.divider()
    st.subheader("📥 Download Debate")
    md_data = export_session_markdown(
        session_id=st.session_state["session_id"],
        persona=st.session_state["persona"],
        topic=st.session_state["debate_topic"],
        transcript=st.session_state["transcript"],
        scores=st.session_state["scores"],
        flags=st.session_state["flags"],
        rewrites=st.session_state["rewrites"],
    )
    st.download_button(
        label="📄 Download Transcript (.md)",
        data=md_data,
        file_name=f"debate_transcript_{st.session_state['session_id'][:8]}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    json_data = export_session(
        session_id=st.session_state["session_id"],
        persona=st.session_state["persona"],
        topic=st.session_state["debate_topic"],
        evidence_items=st.session_state["evidence_items"],
        transcript=st.session_state["transcript"],
        last_scores=st.session_state["scores"],
        last_flags=st.session_state["flags"],
    )
    st.download_button(
        label="📊 Export Session Data (.json)",
        data=json_data,
        file_name=f"debate_session_{st.session_state['session_id'][:8]}.json",
        mime="application/json",
        use_container_width=True,
    )

# ─────────────────────────────────────────────
# LLM CLIENT INITIALIZATION & RESOLUTION
# ─────────────────────────────────────────────

def resolve_gemini_api_key() -> str:
    # 1. Custom key entered in UI sidebar
    if st.session_state.get("custom_api_key"):
        key = st.session_state["custom_api_key"].strip()
        if key:
            return key
    # 2. Environment variable
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key
    # 3. Streamlit Secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            sec_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if sec_key:
                return sec_key
    except Exception:
        pass
    return ""

_resolved_key = resolve_gemini_api_key()
selected_model = st.session_state.get("selected_gemini_model", "gemini-2.0-flash")

if (
    "_llm_instance" not in st.session_state
    or st.session_state.get("_llm_key") != _resolved_key
    or st.session_state.get("_llm_model") != selected_model
):
    if _resolved_key:
        try:
            st.session_state["_llm_instance"] = LLMClient(api_key=_resolved_key, model_name=selected_model)
            st.session_state["_using_mock"] = False
            st.session_state["_llm_error"] = ""
        except Exception as e:
            st.session_state["_llm_instance"] = MockLLMClient()
            st.session_state["_using_mock"] = True
            st.session_state["_llm_error"] = str(e)
    else:
        st.session_state["_llm_instance"] = MockLLMClient()
        st.session_state["_using_mock"] = True
        st.session_state["_llm_error"] = "No API Key provided."
    st.session_state["_llm_key"] = _resolved_key
    st.session_state["_llm_model"] = selected_model

_llm = st.session_state["_llm_instance"]
_using_mock = st.session_state.get("_using_mock", True)

with st.sidebar:
    st.divider()
    st.subheader("🔑 API Key & Model")
    user_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.get("custom_api_key", ""),
        type="password",
        help="Paste your Google Gemini API key here. Get one free at https://aistudio.google.com/app/apikey",
    )
    if user_key_input != st.session_state.get("custom_api_key", ""):
        st.session_state["custom_api_key"] = user_key_input
        st.rerun()

    model_options = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    current_model = st.session_state.get("selected_gemini_model", "gemini-2.0-flash")
    if current_model not in model_options:
        current_model = "gemini-2.0-flash"
    new_model = st.selectbox("Gemini Model", model_options, index=model_options.index(current_model))
    if new_model != st.session_state.get("selected_gemini_model"):
        st.session_state["selected_gemini_model"] = new_model
        st.rerun()

    if _using_mock:
        err_msg = st.session_state.get("_llm_error", "")
        if err_msg and err_msg != "No API Key provided.":
            st.error(f"⚠️ **API Key Error**: {err_msg}")
        st.warning("⚠️ **Demo Mode Active** (Offline Mock LLM).\nProvide a valid Gemini API key above to enable live AI responses.")
    else:
        st.success(f"✅ **Live Gemini Active** ({st.session_state.get('selected_gemini_model', 'gemini-2.0-flash')})")

    st.caption(f"Practice mode: **{st.session_state['practice_mode']}**")
# ─────────────────────────────────────────────
# MAIN LAYOUT — CHAT (LEFT) + DASHBOARD (RIGHT)

col_chat, col_dash = st.columns([3, 2], gap="large")

# ── LEFT: CHAT ──────────────────────────────
with col_chat:
    st.markdown("### 💬 Debate")
    topic_label = st.session_state["debate_topic"] or "Waiting for your first substantive argument"
    st.caption(f"Topic: **{topic_label}**")
    st.caption(f"You are debating against **{st.session_state['persona']}**")
    st.caption(f"Practice mode: **{st.session_state['practice_mode']}**")
    st.info(
        "This app separates the AI stakeholder into a role and a policy orientation. "
        "Choose the role that represents who is speaking, and the orientation that represents their stance on harm reduction vs abstinence. "
        "For example: Clinician + Harm Reduction."
    )
    st.info(
        "Debate practice tip: make a clear claim, support it with evidence or logic, and anticipate one objection. "
        "Use your next turn to strengthen your argument or rebut the AI's position."
    )
    if st.session_state.get("reply_error"):
        st.error(st.session_state["reply_error"])

    chat_container = st.container(height=360)
    with chat_container:
        if not st.session_state["transcript"]:
            st.info("Type your opening argument below to begin the debate.")
        for turn in st.session_state["transcript"]:
            with st.chat_message(turn["role"]):
                st.write(turn["content"])

    user_input = st.chat_input(
        "Your argument…",
        disabled=st.session_state.get("input_locked", False),
    )
    if user_input and not st.session_state.get("input_locked", False):
        st.session_state["pending_user_input"] = user_input
        st.session_state["input_locked"] = True
        st.rerun()

    pending_user_input = st.session_state.get("pending_user_input", "")
    if st.session_state.get("input_locked", False) and not pending_user_input:
        st.session_state["input_locked"] = False
    if pending_user_input:
        user_input = pending_user_input
        st.session_state["pending_user_input"] = ""
        st.session_state["reply_error"] = ""
        # 1. Append user turn
        st.session_state["transcript"].append({"role": "user", "content": user_input})

        # 2. Infer the debate topic from the first real debate turn, and allow
        # a later stronger policy question to replace a weaker inferred topic.
        turn_strength = topic_turn_strength(user_input)
        should_refresh_topic = (
            turn_strength > 0 and (
                not st.session_state["debate_topic"]
                or (
                    st.session_state["topic_source"] != "manual"
                    and not is_valid_topic_label(st.session_state["debate_topic"])
                )
                or (
                    st.session_state["topic_source"] != "manual"
                    and st.session_state["topic_strength"] < turn_strength
                )
                or (
                    st.session_state["topic_source"] != "manual"
                    and is_explicit_topic_switch_turn(user_input)
                )
            )
        )
        if should_refresh_topic or not st.session_state["debate_topic"]:
            inferred_topic = infer_debate_topic(st.session_state["transcript"], _llm)
            st.session_state["debate_topic"] = inferred_topic or user_input[:100]
            st.session_state["topic_source"] = "inferred"
            st.session_state["topic_strength"] = turn_strength or 1

        # 3. Generate AI stakeholder response (transcript already has the latest user turn)
        retrieved_source_hits = []
        should_retrieve_sources = (
            st.session_state["debate_topic"]
            and (is_substantive_debate_turn(user_input) or is_evidence_request(user_input))
        )
        if should_retrieve_sources:
            try:
                retrieved_source_hits = lookup_relevant_source_hits(
                    st.session_state["persona"],
                    st.session_state["debate_topic"],
                    st.session_state["transcript"],
                    force_fallback=True,
                )
            except Exception:
                retrieved_source_hits = []

        # Spin off background research subagent for counter perspectives
        spin_off_research_subagent(
            session_id=st.session_state["session_id"],
            user_claim=user_input,
            persona_id=st.session_state["persona"],
            debate_topic=st.session_state["debate_topic"],
            transcript=st.session_state["transcript"],
        )
        subagent_counter_research = get_synthesized_counter_evidence(st.session_state["session_id"])

        prompt = build_debate_prompt(
            st.session_state["persona"],
            st.session_state["debate_topic"],
            st.session_state["transcript"],
            st.session_state["evidence_items"],
            retrieved_source_hits=retrieved_source_hits,
            subagent_counter_research=subagent_counter_research,
            debate_mode=st.session_state["debate_mode"],
        )

        generation_error = ""
        ai_reply = ""
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                thinking_placeholder.write("Thinking...")
                with st.spinner("Thinking..."):
                    try:
                        ai_reply = _llm.generate(prompt, temperature=0.5, max_output_tokens=1500)
                    except Exception as e:
                        generation_error = str(e)
                        ai_reply = ""
                        # Retry with a more compact prompt if Gemini fails.
                        try:
                            fallback_prompt = build_debate_prompt(
                                st.session_state["persona"],
                                st.session_state["debate_topic"],
                                st.session_state["transcript"],
                                st.session_state["evidence_items"],
                                retrieved_source_hits=retrieved_source_hits,
                                include_source_catalog=False,
                                debate_mode=st.session_state["debate_mode"],
                            )
                            ai_reply = _llm.generate(fallback_prompt, temperature=0.5, max_output_tokens=1200)
                        except Exception:
                            ai_reply = ""
                        if not ai_reply:
                            st.session_state["reply_error"] = (
                                "Gemini request failed: "
                                f"{generation_error or 'the service returned no text'}. "
                                "Check Manage app > Logs for the full error."
                            )
                if ai_reply:
                    thinking_placeholder.write(ai_reply)
                elif generation_error:
                    thinking_placeholder.write("I ran into an error generating that reply.")

        # 4. Append AI reply BEFORE scoring (task requirement)
        if ai_reply:
            st.session_state["transcript"].append({"role": "assistant", "content": ai_reply})

        # 5. Score substantive turns only after a topic has been identified
        turn_scores = None
        turn_flags = []
        if generation_error:
            st.session_state["rewrites"] = []
            st.session_state["reflections"] = []
            st.session_state["flags"] = []
            st.session_state["scoring_error"] = ""
            st.session_state["scoring_note"] = (
                "The AI could not finish its reply on that turn, so coaching feedback was skipped."
            )
        elif st.session_state["debate_topic"] and is_substantive_debate_turn(user_input):
            try:
                scoring: ScoringOutput = score_turn_with_llm(
                    st.session_state["transcript"], user_input, _llm
                )
                st.session_state["scores"] = {
                    "ethos": scoring.scores.ethos,
                    "logos": scoring.scores.logos,
                    "pathos": scoring.scores.pathos,
                }
                st.session_state["scoring_rationale"] = scoring.rationale
                st.session_state["rewrites"] = scoring.rewrite_suggestions
                st.session_state["reflections"] = scoring.reflection_prompts
                st.session_state["flags"] = [
                    {"flagged": f, "suggestion": f} for f in scoring.ethics_flags
                ]
                st.session_state["scoring_error"] = ""
                st.session_state["scoring_note"] = ""
                turn_scores = dict(st.session_state["scores"])
                turn_flags = list(st.session_state["flags"])
            except Exception as e:
                st.session_state["scoring_error"] = str(e)
                st.session_state["scoring_note"] = ""
        else:
            st.session_state["rewrites"] = []
            st.session_state["reflections"] = []
            st.session_state["flags"] = []
            st.session_state["scoring_error"] = ""
            if not st.session_state["debate_topic"]:
                st.session_state["scoring_note"] = (
                    "The last message did not establish a clear debate topic yet, so the AI asked for clarification."
                )
            else:
                st.session_state["scoring_note"] = (
                    "The last message was not scored because it did not present a substantive policy argument."
                )

        # 6. Persist log
        log_entry = {
            "session_id": st.session_state["session_id"],
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "persona": st.session_state["persona"],
            "topic": st.session_state["debate_topic"],
            "topic_source": st.session_state["topic_source"],
            "user_turn": user_input,
            "ai_turn": ai_reply,
            "response_error": generation_error,
            "turn_scored": turn_scores is not None,
            "scores": turn_scores,
            "flags": turn_flags,
        }
        append_log(log_entry)
        st.session_state["input_locked"] = False
        st.rerun()

# ── RIGHT: DASHBOARD ─────────────────────────
with col_dash:
    st.subheader("📊 Coaching Dashboard")

    # Scoring error banner
    if st.session_state.get("scoring_error"):
        st.error(f"⚠️ Scoring failed: {st.session_state['scoring_error']}")
    elif st.session_state.get("scoring_note"):
        st.info(st.session_state["scoring_note"])

    # ── Radar chart ──
    cats = ["Ethos", "Logos", "Pathos"]
    vals = [
        st.session_state["scores"]["ethos"],
        st.session_state["scores"]["logos"],
        st.session_state["scores"]["pathos"],
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        line_color="#0d3b6e",
        fillcolor="rgba(13,59,110,0.2)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=260,
    )
    st.plotly_chart(fig, width="stretch")

    # Score bars
    score_cols = st.columns(3)
    labels = ["Ethos", "Logos", "Pathos"]
    colors = ["#0d3b6e", "#c8102e", "#63666a"]
    for col, label, key, color in zip(score_cols, labels, ["ethos", "logos", "pathos"], colors):
        with col:
            v = st.session_state["scores"][key]
            st.metric(label, f"{v} / 5")

    st.divider()

    # ── Scoring rationale ──
    if st.session_state["scoring_rationale"]:
        st.markdown("**🧠 Why these scores?**")
        for label, key in [("Ethos", "ethos"), ("Logos", "logos"), ("Pathos", "pathos")]:
            rationale = st.session_state["scoring_rationale"].get(key, "No rationale provided.")
            st.markdown(f"- **{label}:** {rationale}")
        st.divider()

    # ── Coaching focus ──
    tips = []
    if st.session_state["scores"]["logos"] < 3:
        tips.append("Strengthen your logic by adding an explicit reason or evidence link.")
    if st.session_state["scores"]["ethos"] < 3:
        tips.append("Build credibility with source-aware language and person-first framing.")
    if st.session_state["scores"]["pathos"] < 3:
        tips.append("Connect the argument to people, values, or real-world consequences.")
    if not tips:
        tips.append("Your last turn was balanced; keep your reasoning clear and your evidence grounded.")
    st.markdown("**🎯 Coaching focus**")
    for tip in tips:
        st.markdown(f"- {tip}")
    st.divider()

    # ── Rewrite suggestions ──
    if st.session_state["rewrites"]:
        st.markdown("**✍️ Rewrite Suggestions**")
        for rw in st.session_state["rewrites"]:
            st.markdown(f'<div class="rewrite-box">{rw}</div>', unsafe_allow_html=True)

    # ── Reflection prompts ──
    if st.session_state["reflections"]:
        st.markdown("**🪞 Reflection Prompts**")
        for rf in st.session_state["reflections"]:
            st.markdown(f'<div class="reflect-box">{rf}</div>', unsafe_allow_html=True)

    # ── Ethics / language flags ──
    if st.session_state["flags"]:
        st.markdown("**⚠️ Language Flags**")
        for fl in st.session_state["flags"]:
            st.markdown(
                f'<div class="flag-box">🚩 <b>"{fl["flagged"]}"</b> — {fl["suggestion"]}</div>',
                unsafe_allow_html=True,
            )
    elif st.session_state["transcript"]:
        st.success("✅ No language flags detected.")
