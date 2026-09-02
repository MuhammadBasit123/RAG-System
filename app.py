"""
RAG Question-Answering System — Streamlit Web Application (Ultra 3D Premium UI)

Features:
  Tab 1 — Ask a Question  (single strategy, citations displayed)
  Tab 2 — Compare Strategies  (side-by-side A vs B)
  Tab 3 — Evaluation Results  (pre-run comparison table)

Run:
    streamlit run app.py
"""

from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Deep Learning RAG Q&A · Intelligent System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Ultra Premium 3D CSS ─────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

/* ══ ROOT ════════════════════════════════════════════ */
:root {
    --purple-500: #a855f7;
    --blue-500:   #3b82f6;
    --cyan-400:   #22d3ee;
    --teal-400:   #2dd4bf;
    --green-400:  #4ade80;
    --rose-400:   #fb7185;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ══ ANIMATED DEEP SPACE BACKGROUND ════════════════════ */
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 10% 20%, rgba(139,92,246,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(59,130,246,0.16) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(16,185,129,0.08) 0%, transparent 60%),
        linear-gradient(160deg, #070713 0%, #0d0b20 30%, #08111e 70%, #050d18 100%) !important;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}
.stApp::before {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
    animation: orbFloat1 18s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -150px; right: -150px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(59,130,246,0.10) 0%, transparent 70%);
    border-radius: 50%;
    animation: orbFloat2 22s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes orbFloat1 {
    0%,100% { transform: translate(0,0) scale(1); }
    33%      { transform: translate(60px,-80px) scale(1.15); }
    66%      { transform: translate(-40px,50px) scale(0.9); }
}
@keyframes orbFloat2 {
    0%,100% { transform: translate(0,0) scale(1); }
    40%      { transform: translate(-70px,60px) scale(1.2); }
    70%      { transform: translate(50px,-40px) scale(0.85); }
}

/* ══ FLOATING PARTICLES ══════════════════════════════ */
.particle-field {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none; z-index: 0; overflow: hidden;
}
.particle {
    position: absolute; border-radius: 50%;
    animation: floatParticle linear infinite; opacity: 0;
}
.particle:nth-child(1)  { width:3px;height:3px;background:rgba(168,85,247,0.7);left:10%;animation-duration:12s;animation-delay:0s; }
.particle:nth-child(2)  { width:2px;height:2px;background:rgba(96,165,250,0.6);left:25%;animation-duration:15s;animation-delay:2s; }
.particle:nth-child(3)  { width:4px;height:4px;background:rgba(34,211,238,0.5);left:40%;animation-duration:10s;animation-delay:4s; }
.particle:nth-child(4)  { width:2px;height:2px;background:rgba(168,85,247,0.8);left:60%;animation-duration:18s;animation-delay:1s; }
.particle:nth-child(5)  { width:3px;height:3px;background:rgba(45,212,191,0.6);left:75%;animation-duration:14s;animation-delay:6s; }
.particle:nth-child(6)  { width:2px;height:2px;background:rgba(96,165,250,0.5);left:85%;animation-duration:11s;animation-delay:3s; }
.particle:nth-child(7)  { width:5px;height:5px;background:rgba(251,191,36,0.3);left:50%;animation-duration:20s;animation-delay:8s; }
.particle:nth-child(8)  { width:2px;height:2px;background:rgba(168,85,247,0.6);left:15%;animation-duration:16s;animation-delay:5s; }
.particle:nth-child(9)  { width:3px;height:3px;background:rgba(34,211,238,0.7);left:90%;animation-duration:13s;animation-delay:7s; }
.particle:nth-child(10) { width:2px;height:2px;background:rgba(45,212,191,0.5);left:35%;animation-duration:17s;animation-delay:9s; }
@keyframes floatParticle {
    0%   { transform: translateY(100vh) scale(0); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { transform: translateY(-100px) scale(1.5); opacity: 0; }
}

/* ══ SIDEBAR ═════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(10,8,30,0.85) !important;
    border-right: 1px solid rgba(139,92,246,0.25) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    box-shadow: inset -1px 0 0 rgba(139,92,246,0.15), 4px 0 40px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #a855f7, #3b82f6, #22d3ee);
    border-radius: 0 0 4px 4px;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.88) !important; }
.sidebar-brand {
    text-align: center;
    padding: 1.2rem 0 1rem;
    margin-bottom: 0.5rem;
}
.sidebar-brain-icon {
    font-size: 3rem; display: block; margin-bottom: 0.4rem;
    filter: drop-shadow(0 0 20px rgba(168,85,247,0.8));
    animation: brainPulse 3s ease-in-out infinite;
}
@keyframes brainPulse {
    0%,100% { filter: drop-shadow(0 0 15px rgba(168,85,247,0.7)); transform: scale(1); }
    50%      { filter: drop-shadow(0 0 30px rgba(96,165,250,0.9)); transform: scale(1.08); }
}
.sidebar-brand-text {
    font-weight: 800; font-size: 1.05rem;
    background: linear-gradient(135deg, #c084fc, #60a5fa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: 0.04em;
}
.sidebar-brand-sub {
    font-size: 0.7rem; color: rgba(255,255,255,0.4) !important;
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.15rem;
}
.sidebar-info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px; padding: 0.9rem 1rem; margin: 0.5rem 0;
    position: relative; overflow: hidden;
}
.sidebar-info-card::before {
    content: '';
    position: absolute; top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, #a855f7, #3b82f6); border-radius: 3px 0 0 3px;
}
.sidebar-info-card h4 {
    font-size: 0.8rem !important; font-weight: 700 !important;
    color: #c084fc !important; margin: 0 0 0.5rem !important; letter-spacing: 0.05em;
}
.sidebar-info-card ul { margin: 0; padding-left: 1rem; }
.sidebar-info-card li {
    font-size: 0.78rem !important; color: rgba(255,255,255,0.6) !important;
    margin-bottom: 0.2rem; line-height: 1.5;
}
.threshold-display {
    background: linear-gradient(135deg, rgba(168,85,247,0.12), rgba(59,130,246,0.08));
    border: 1px solid rgba(168,85,247,0.3); border-radius: 10px;
    padding: 0.7rem 1rem; text-align: center; margin-top: 0.5rem;
    font-size: 0.85rem; font-family: 'JetBrains Mono', monospace;
    color: #c084fc; letter-spacing: 0.05em;
}

/* ══ 3D HERO HEADER ══════════════════════════════════ */
.rag-header {
    position: relative;
    background:
        radial-gradient(ellipse 60% 80% at 30% 40%, rgba(139,92,246,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 70% 60%, rgba(59,130,246,0.14) 0%, transparent 55%),
        rgba(255,255,255,0.028);
    border: 1px solid rgba(255,255,255,0.10);
    border-top: 1px solid rgba(255,255,255,0.18);
    border-radius: 28px;
    padding: 3.5rem 3rem 3rem;
    margin-bottom: 2rem; text-align: center;
    backdrop-filter: blur(32px); -webkit-backdrop-filter: blur(32px);
    overflow: hidden;
    box-shadow: 0 2px 0 rgba(255,255,255,0.06) inset, 0 40px 80px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.05);
}
.rag-header::before {
    content: '';
    position: absolute; inset: 0;
    background-image: linear-gradient(rgba(168,85,247,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(168,85,247,0.06) 1px, transparent 1px);
    background-size: 40px 40px; border-radius: 28px;
    animation: gridShift 20s linear infinite; pointer-events: none;
}
.rag-header::after {
    content: '';
    position: absolute; top: 0; left: 10%; right: 10%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #a855f7, #3b82f6, #22d3ee, transparent);
    border-radius: 100px; animation: shimmerLine 3s ease-in-out infinite;
}
@keyframes gridShift { 0% { background-position: 0 0; } 100% { background-position: 40px 40px; } }
@keyframes shimmerLine { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
.rag-header .header-brain {
    font-size: 4rem; display: inline-block; margin-bottom: 0.8rem;
    filter: drop-shadow(0 0 30px rgba(168,85,247,0.9));
    animation: heroIconFloat 4s ease-in-out infinite;
    position: relative; z-index: 1;
}
@keyframes heroIconFloat {
    0%,100% { transform: translateY(0) rotate(-2deg); }
    50%      { transform: translateY(-10px) rotate(2deg); }
}
.rag-header h1 {
    font-size: 3rem !important; font-weight: 900 !important;
    background: linear-gradient(135deg, #e9d5ff 0%, #93c5fd 40%, #5eead4 70%, #86efac 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0 0 0.7rem !important; line-height: 1.15 !important;
    letter-spacing: -0.02em; position: relative; z-index: 1;
}
.rag-header .subtitle {
    color: rgba(255,255,255,0.5); font-size: 1rem;
    margin: 0 0 0.5rem; letter-spacing: 0.04em;
    position: relative; z-index: 1;
}
.header-badges {
    display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap;
    margin-top: 1rem; position: relative; z-index: 1;
}
.header-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.3rem 0.9rem; border-radius: 100px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; backdrop-filter: blur(8px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.header-badge:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
.hb-purple { background: rgba(168,85,247,0.18); border: 1px solid rgba(168,85,247,0.4); color: #e9d5ff; }
.hb-blue   { background: rgba(59,130,246,0.18);  border: 1px solid rgba(59,130,246,0.4);  color: #bfdbfe; }
.hb-cyan   { background: rgba(34,211,238,0.15);  border: 1px solid rgba(34,211,238,0.4);  color: #a5f3fc; }
.hb-green  { background: rgba(74,222,128,0.12);  border: 1px solid rgba(74,222,128,0.35); color: #bbf7d0; }

/* ══ GLASS CARDS (answer / unknown / source) ════════════ */
.answer-card {
    background: radial-gradient(ellipse 60% 80% at 0% 0%, rgba(74,222,128,0.08) 0%, transparent 60%), rgba(255,255,255,0.038);
    border: 1px solid rgba(74,222,128,0.25); border-top: 1px solid rgba(74,222,128,0.35);
    border-radius: 20px; padding: 1.8rem 2rem; margin: 1rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 2px 0 rgba(74,222,128,0.05) inset, 0 20px 60px rgba(0,0,0,0.4), 0 0 30px rgba(74,222,128,0.06);
    animation: cardReveal 0.5s cubic-bezier(0.22,0.61,0.36,1) both;
    position: relative; overflow: hidden;
}
.answer-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%;
    background: linear-gradient(180deg, #4ade80, #22d3ee);
}
.answer-card .card-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #86efac; margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.answer-card .card-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(74,222,128,0.3), transparent);
}
.answer-card .card-body { color: rgba(255,255,255,0.93); font-size: 1.08rem; line-height: 1.8; padding-left: 0.5rem; }

.unknown-card {
    background: radial-gradient(ellipse 60% 80% at 0% 0%, rgba(251,113,133,0.10) 0%, transparent 60%), rgba(255,255,255,0.025);
    border: 1px solid rgba(251,113,133,0.28); border-top: 1px solid rgba(251,113,133,0.40);
    border-radius: 20px; padding: 1.8rem 2rem; margin: 1rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 2px 0 rgba(251,113,133,0.05) inset, 0 20px 60px rgba(0,0,0,0.4), 0 0 30px rgba(251,113,133,0.08);
    animation: cardReveal 0.5s cubic-bezier(0.22,0.61,0.36,1) both;
    position: relative; overflow: hidden;
}
.unknown-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%;
    background: linear-gradient(180deg, #fb7185, #f43f5e);
}
.unknown-card .card-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #fda4af; margin-bottom: 0.8rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.unknown-card .card-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(251,113,133,0.3), transparent);
}
.unknown-card .card-body { color: rgba(255,200,210,0.92); font-size: 1rem; line-height: 1.75; padding-left: 0.5rem; }

.source-card {
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(96,165,250,0.18); border-radius: 14px;
    padding: 1rem 1.3rem; margin: 0.5rem 0;
    backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    animation: cardReveal 0.6s cubic-bezier(0.22,0.61,0.36,1) both;
    transition: transform 0.2s ease, border-color 0.2s ease;
    position: relative; overflow: hidden;
}
.source-card:hover { transform: translateX(4px); border-color: rgba(96,165,250,0.35); }
.source-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
    background: linear-gradient(180deg, #60a5fa, #818cf8);
}
.source-card .source-meta { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 0.6rem; padding-left: 0.3rem; }
.source-card .source-text {
    color: rgba(255,255,255,0.52); font-size: 0.87rem; line-height: 1.65;
    font-style: italic; padding-left: 0.3rem;
    border-left: 1px solid rgba(255,255,255,0.08); margin-left: 0.3rem;
}
.score-bar-track { height: 4px; background: rgba(255,255,255,0.06); border-radius: 4px; margin-top: 0.7rem; overflow: hidden; }
.score-bar {
    height: 4px; border-radius: 4px;
    background: linear-gradient(90deg, #818cf8, #a855f7, #22d3ee);
    box-shadow: 0 0 10px rgba(168,85,247,0.6);
    animation: scoreGrow 0.8s cubic-bezier(0.22,0.61,0.36,1) both 0.2s;
    transform-origin: left;
}
@keyframes scoreGrow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* ══ PILL BADGES ═════════════════════════════════════ */
.badge {
    display: inline-flex; align-items: center; gap: 0.2rem;
    padding: 0.2rem 0.65rem; border-radius: 100px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
    font-family: 'JetBrains Mono', monospace;
}
.badge-purple { background: rgba(168,85,247,0.18); border: 1px solid rgba(168,85,247,0.4); color: #e9d5ff; }
.badge-blue   { background: rgba(59,130,246,0.15);  border: 1px solid rgba(59,130,246,0.35); color: #bfdbfe; }
.badge-green  { background: rgba(74,222,128,0.12);  border: 1px solid rgba(74,222,128,0.3);  color: #bbf7d0; }
.badge-red    { background: rgba(251,113,133,0.15); border: 1px solid rgba(251,113,133,0.35);color: #fecdd3; }

/* ══ SECTION TITLES ══════════════════════════════════ */
.section-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: rgba(255,255,255,0.35);
    margin: 1.5rem 0 0.7rem; display: flex; align-items: center; gap: 0.6rem;
}
.section-title::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0.1), transparent); }

/* ══ STRATEGY COLUMNS ════════════════════════════════ */
.strat-col-a, .strat-col-b {
    border-radius: 18px; padding: 1.2rem 1.5rem; text-align: center;
    margin-bottom: 1.2rem; position: relative; overflow: hidden;
    backdrop-filter: blur(16px); box-shadow: 0 16px 48px rgba(0,0,0,0.35);
}
.strat-col-a {
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(168,85,247,0.2) 0%, transparent 70%), rgba(255,255,255,0.04);
    border: 1px solid rgba(168,85,247,0.3); border-top: 1px solid rgba(168,85,247,0.5);
}
.strat-col-b {
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(59,130,246,0.2) 0%, transparent 70%), rgba(255,255,255,0.04);
    border: 1px solid rgba(59,130,246,0.3); border-top: 1px solid rgba(59,130,246,0.5);
}
.strat-col-a::before { content: ''; position: absolute; top: 0; left: 10%; right: 10%; height: 2px; background: linear-gradient(90deg, transparent, #a855f7, transparent); }
.strat-col-b::before { content: ''; position: absolute; top: 0; left: 10%; right: 10%; height: 2px; background: linear-gradient(90deg, transparent, #3b82f6, transparent); }
.strat-col-a h3 { color:#e9d5ff; margin:0; font-size:1.15rem; font-weight:700; }
.strat-col-b h3 { color:#bfdbfe; margin:0; font-size:1.15rem; font-weight:700; }
.strat-col-a p, .strat-col-b p { color: rgba(255,255,255,0.45); font-size: 0.8rem; margin: 0.35rem 0 0; }
.strat-icon { font-size: 1.8rem; display: block; margin-bottom: 0.4rem; }

/* ══ TABS ════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 14px !important; padding: 0.3rem !important;
    border: 1px solid rgba(255,255,255,0.07) !important; gap: 0.2rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important; color: rgba(255,255,255,0.5) !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    padding: 0.6rem 1.2rem !important; transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(255,255,255,0.06) !important; color: rgba(255,255,255,0.85) !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(168,85,247,0.25), rgba(59,130,246,0.18)) !important;
    color: #fff !important; box-shadow: 0 4px 16px rgba(168,85,247,0.2) !important;
    border: 1px solid rgba(168,85,247,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: transparent !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ══ INPUTS & BUTTONS ════════════════════════════════ */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(168,85,247,0.28) !important;
    border-radius: 12px !important; color: #fff !important;
    font-size: 1rem !important; padding: 0.85rem 1.1rem !important;
    transition: all 0.25s ease !important; backdrop-filter: blur(8px) !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(168,85,247,0.6) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.12), 0 0 30px rgba(168,85,247,0.15) !important;
    background: rgba(255,255,255,0.07) !important;
}
.stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.3) !important; }
.stButton > button {
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 0.9rem !important; letter-spacing: 0.04em !important;
    padding: 0.65rem 1.4rem !important; border: none !important;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
    position: relative !important; overflow: hidden !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.4), 0 0 0 1px rgba(255,255,255,0.08) inset !important;
    color: #fff !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 8px 36px rgba(124,58,237,0.55), 0 0 0 1px rgba(255,255,255,0.1) inset !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) scale(0.99) !important; }
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.82rem !important; font-weight: 500 !important; padding: 0.45rem 0.85rem !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: rgba(168,85,247,0.12) !important;
    border-color: rgba(168,85,247,0.35) !important;
    color: #e9d5ff !important; transform: translateY(-1px) !important;
}

/* ══ SLIDER ══════════════════════════════════════════ */
.stSlider > div > div > div > div { background: linear-gradient(90deg, #7c3aed, #2563eb) !important; }
.stSlider > div > div > div > div > div { background: #fff !important; box-shadow: 0 0 0 3px rgba(168,85,247,0.4) !important; }

/* ══ METRIC CARDS ════════════════════════════════════ */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important; padding: 1rem 1.2rem !important;
    backdrop-filter: blur(12px) !important; box-shadow: 0 8px 32px rgba(0,0,0,0.25) !important;
    transition: transform 0.2s ease !important;
}
[data-testid="metric-container"]:hover { transform: translateY(-2px) !important; }
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.45) !important; font-size: 0.75rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #fff !important; font-size: 1.6rem !important; font-weight: 800 !important;
}

/* ══ DATAFRAME ═══════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 16px !important; overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: 0 16px 48px rgba(0,0,0,0.35) !important;
}

/* ══ EXPANDER ════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important; border: 1px solid rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.75) !important; font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}

/* ══ DIVIDERS ════════════════════════════════════════ */
hr {
    border: none !important; height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.3), rgba(59,130,246,0.3), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ══ EXAMPLE SECTION HEADERS ═════════════════════════ */
.example-section-header {
    display: flex; align-items: center; gap: 0.7rem; margin: 1.2rem 0 0.7rem;
}
.example-section-header .dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #4ade80; box-shadow: 0 0 10px rgba(74,222,128,0.7);
    animation: dotPulse 2s ease-in-out infinite;
}
.example-section-header .dot.red { background: #fb7185; box-shadow: 0 0 10px rgba(251,113,133,0.7); }
.example-section-header span {
    font-size: 0.82rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: rgba(255,255,255,0.5);
}
@keyframes dotPulse { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.4); opacity: 0.7; } }

/* ══ WINNER BANNER ═══════════════════════════════════ */
.winner-banner {
    text-align: center; padding: 1.2rem; border-radius: 16px;
    margin: 1rem 0; font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.05em; animation: scaleIn 0.4s ease both;
}
.winner-a { background: radial-gradient(ellipse at center, rgba(168,85,247,0.2), transparent); border: 1px solid rgba(168,85,247,0.4); color: #e9d5ff; }
.winner-b { background: radial-gradient(ellipse at center, rgba(59,130,246,0.2), transparent);  border: 1px solid rgba(59,130,246,0.4);  color: #bfdbfe; }
.winner-tie { background: radial-gradient(ellipse at center, rgba(251,191,36,0.12), transparent); border: 1px solid rgba(251,191,36,0.3); color: #fde68a; }

/* ══ ANIMATIONS ══════════════════════════════════════ */
@keyframes cardReveal {
    from { opacity: 0; transform: perspective(600px) translateY(20px) rotateX(-5deg); }
    to   { opacity: 1; transform: perspective(600px) translateY(0) rotateX(0deg); }
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
@keyframes scaleIn { from { opacity: 0; transform: scale(0.92); } to { opacity: 1; transform: scale(1); } }

/* ══ SPINNER ═════════════════════════════════════════ */
.stSpinner > div { border-top-color: #a855f7 !important; }

/* ══ ALERTS ══════════════════════════════════════════ */
.stAlert {
    background: rgba(255,255,255,0.04) !important; border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.09) !important; backdrop-filter: blur(12px) !important;
}

/* ══ RADIO BUTTONS ═══════════════════════════════════ */
.stRadio > div { gap: 0.5rem; }
.stRadio label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; padding: 0.5rem 0.9rem !important;
    transition: all 0.2s ease !important; cursor: pointer !important;
}
.stRadio label:hover { border-color: rgba(168,85,247,0.35) !important; background: rgba(168,85,247,0.08) !important; }

/* ══ HIDE STREAMLIT CHROME ═══════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }

/* ══ SCROLLBAR ═══════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(168,85,247,0.4); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(168,85,247,0.65); }

/* ══ GLASS CARD (generic) ════════════════════════════ */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09); border-top: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px; padding: 1.8rem 2rem; margin: 1rem 0;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 2px 0 rgba(255,255,255,0.05) inset, 0 20px 60px rgba(0,0,0,0.4);
    animation: cardReveal 0.5s cubic-bezier(0.22,0.61,0.36,1) both;
    position: relative; overflow: hidden;
}
.glass-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
}
</style>

<!-- Floating particles -->
<div class="particle-field">
  <div class="particle"></div><div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div><div class="particle"></div>
  <div class="particle"></div><div class="particle"></div><div class="particle"></div>
  <div class="particle"></div>
</div>
""",
    unsafe_allow_html=True,
)


# ─── Resource caching ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _preload_model():
    """Warm up the fastembed model once per session."""
    from src.embedder import get_model
    get_model()
    return True


# ─── Render helpers ───────────────────────────────────────────────────────────

def _badge(text: str, color: str = "purple") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'


def render_answer(result: dict) -> None:
    """Render an answer card + source citations."""
    gated     = result.get("gated", False)
    answer    = result.get("answer", "")
    sources   = result.get("sources", [])
    top_score = result.get("top_score", 0.0)
    strategy  = result.get("strategy", "?")

    label_extra = (
        f'<span class="badge badge-purple">Strategy {strategy}</span>'
        f'<span class="badge badge-{"green" if top_score >= 0.35 else "red"}">'   
        f'\U0001f3af {top_score:.3f}</span>'
    )

    if gated:
        st.markdown(
            f"""
            <div class="unknown-card">
                <div class="card-label">\u26d4 &nbsp;No Relevant Passage Found &nbsp; {label_extra}</div>
                <div class="card-body">
                    The similarity score (<strong>{top_score:.3f}</strong>) fell below the
                    threshold gate \u2014 the system does not have enough information to answer
                    this question reliably.<br><br>
                    <strong style="color:#fda4af;">I don't know based on the provided documents.</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="answer-card">
                <div class="card-label">\u2705 &nbsp;Answer &nbsp; {label_extra}</div>
                <div class="card-body">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if sources:
        st.markdown(
            '<div class="section-title">\U0001f4da &nbsp;Retrieved Passages \u2014 Citations</div>',
            unsafe_allow_html=True,
        )
        for i, r in enumerate(sources, 1):
            chunk     = r["chunk"]
            score     = r["score"]
            preview   = chunk["text"][:300].replace("\n", " ").replace('"', "&quot;")
            score_pct = min(int(score * 100), 100)

            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-meta">
                        {_badge(f"#{i}", "purple")}
                        {_badge("\U0001f4c4 " + chunk["source"], "blue")}
                        {_badge("\U0001f4d6 p." + str(chunk["page"]), "blue")}
                        {_badge(f"\U0001f3af {score:.3f}", "green")}
                    </div>
                    <div class="source-text">"{preview}\u2026"</div>
                    <div class="score-bar-track">
                        <div class="score-bar" style="width:{score_pct}%"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )



# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-brain-icon">\U0001f9e0</span>
            <div class="sidebar-brand-text">RAG Intelligence</div>
            <div class="sidebar-brand-sub">Deep Learning Q&amp;A</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'
        'color:rgba(255,255,255,0.35);margin-bottom:0.5rem;">\u2699\ufe0f Configuration</div>',
        unsafe_allow_html=True,
    )

    strategy_opt = st.radio(
        "Chunking Strategy",
        ["\u26a1 A \u2014 Fixed-Size (no overlap)", "\U0001f9ec B \u2014 Sentence-Aware (with overlap)"],
        help="Choose which chunking strategy is used for retrieval.",
        label_visibility="collapsed",
    )
    strategy: str = "A" if "A \u2014" in strategy_opt else "B"

    st.markdown(
        '<div style="font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;'
        'color:rgba(255,255,255,0.35);margin:1rem 0 0.5rem;">\U0001f39b\ufe0f Retrieval Parameters</div>',
        unsafe_allow_html=True,
    )

    top_k: int = st.slider("Top-K passages", 1, 5, 3)
    threshold: float = st.slider(
        "Similarity threshold",
        0.10, 0.90, 0.35, 0.05,
        help="Below this score the system returns 'I don't know' without calling the LLM.",
    )

    st.markdown("---")

    if strategy == "A":
        st.markdown(
            """
            <div class="sidebar-info-card">
                <h4>\u26a1 Strategy A \u2014 Fixed-Size</h4>
                <ul>
                    <li>Chunk size: <strong>800 characters</strong></li>
                    <li>No overlap</li>
                    <li>No sentence awareness</li>
                    <li>Fast &amp; simple</li>
                </ul>
                <p style="font-size:0.72rem;color:rgba(255,255,255,0.3);margin:0.5rem 0 0;font-style:italic;">
                    May split mid-sentence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="sidebar-info-card">
                <h4>\U0001f9ec Strategy B \u2014 Sentence-Aware</h4>
                <ul>
                    <li>Target: <strong>~600 characters</strong></li>
                    <li>Overlap: <strong>150 characters</strong></li>
                    <li>NLTK sentence tokenisation</li>
                    <li>Better boundary coherence</li>
                </ul>
                <p style="font-size:0.72rem;color:rgba(255,255,255,0.3);margin:0.5rem 0 0;font-style:italic;">
                    Preserves semantic context.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="threshold-display">\U0001f512 Gate threshold: {threshold:.2f}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center;padding:0.5rem 0;">
            <div style="font-size:0.68rem;color:rgba(255,255,255,0.2);letter-spacing:0.06em;text-transform:uppercase;">
                Powered by FAISS \u00b7 FastEmbed \u00b7 Groq
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Hero Header ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="rag-header">
        <div class="header-brain">\U0001f9e0</div>
        <h1>Deep Learning RAG Q&amp;A System</h1>
        <p class="subtitle">
            Retrieval-Augmented Generation &nbsp;\u00b7&nbsp;
            Two Chunking Strategies &nbsp;\u00b7&nbsp;
            Grounded Answers with Full Citations
        </p>
        <div class="header-badges">
            <span class="header-badge hb-purple">\U0001f52c FAISS Vector Search</span>
            <span class="header-badge hb-blue">\u26a1 FastEmbed</span>
            <span class="header-badge hb-cyan">\U0001f310 Groq LLM</span>
            <span class="header-badge hb-green">\U0001f4da Deep Learning Corpus</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pre-load the model in background
with st.spinner("\U0001f504 Loading embedding model (once per session)..."):
    _preload_model()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    ["\U0001f4ac  Ask a Question", "\u2696\ufe0f  Compare Strategies", "\U0001f4ca  Evaluation Results"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ask a Question
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    query: str = st.text_input(
        "Your question:",
        placeholder="e.g. What is backpropagation?  ·  How do transformers work?  ·  What is gradient descent?",
        key="main_query",
        label_visibility="collapsed",
    )

    ask_col, _ = st.columns([1, 7])
    with ask_col:
        ask_btn = st.button("🔍  Ask RAG", type="primary", use_container_width=True)

    if ask_btn:
        if not query.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Retrieving passages and generating answer..."):
                try:
                    from src.pipeline import run_rag
                    result = run_rag(
                        query,
                        strategy=strategy,
                        top_k=top_k,
                        threshold=threshold,
                    )
                    st.markdown("---")
                    render_answer(result)
                except FileNotFoundError:
                    st.error(
                        "⚠️  Indices not found. Run the setup commands:\n\n"
                        "```bash\n"
                        "python download_corpus.py\n"
                        "python build_index.py\n"
                        "```"
                    )
                except ValueError as exc:
                    st.error(f"⚠️  Configuration error:\n\n{exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # Example buttons
    st.markdown("---")
    st.markdown("**💡 In-corpus questions** (should get an answer):")
    ex_in = [
        "What is deep learning?",
        "How does backpropagation work?",
        "How do convolutional neural networks work?",
        "What is a recurrent neural network?",
        "What is gradient descent?",
        "What is transfer learning?",
        "What is a generative adversarial network?",
        "What is natural language processing?",
        "What is the vanishing gradient problem?",
    ]
    cols = st.columns(3)
    for i, ex in enumerate(ex_in):
        with cols[i % 3]:
            if st.button(ex, key=f"in_{i}", use_container_width=True):
                st.session_state["main_query"] = ex
                st.rerun()

    st.markdown("**🚫 Out-of-corpus questions** (should return *I don't know*):")
    ex_out = [
        "What is the current price of Bitcoin?",
        "How do you make spaghetti carbonara?",
        "Who won the FIFA World Cup in 2022?",
    ]
    cols2 = st.columns(3)
    for i, ex in enumerate(ex_out):
        with cols2[i]:
            if st.button(ex, key=f"out_{i}", use_container_width=True):
                st.session_state["main_query"] = ex
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Compare Strategies
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown(
        '<div style="font-size:0.88rem;color:rgba(255,255,255,0.45);margin-bottom:1rem;">'
        'Run the same question through <strong style="color:rgba(255,255,255,0.75);">both strategies simultaneously</strong>'
        ' \u2014 see how Fixed-Size vs Sentence-Aware chunking affects retrieval quality.'
        '</div>',
        unsafe_allow_html=True,
    )

    cq: str = st.text_input(
        "Question to compare:",
        placeholder="\u2726  e.g. What is the transformer architecture?  \u00b7  Explain overfitting",
        key="cmp_query",
        label_visibility="collapsed",
    )

    cmp_btn = st.button("\u2696\ufe0f  Run Head-to-Head Comparison", type="primary")

    if cmp_btn:
        if not cq.strip():
            st.warning("\u26a0\ufe0f  Please enter a question.")
        else:
            from src.pipeline import run_rag

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown(
                    '<div class="strat-col-a">'
                    '<span class="strat-icon">\u26a1</span>'
                    "<h3>Strategy A</h3>"
                    "<p>Fixed-Size \u00b7 800 chars \u00b7 No Overlap</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Running Strategy A..."):
                    try:
                        res_a = run_rag(cq, strategy="A", top_k=top_k, threshold=threshold)
                        render_answer(res_a)
                    except Exception as exc:
                        st.error(str(exc))
                        res_a = None

            with col_b:
                st.markdown(
                    '<div class="strat-col-b">'
                    '<span class="strat-icon">\U0001f9ec</span>'
                    "<h3>Strategy B</h3>"
                    "<p>Sentence-Aware \u00b7 ~600 chars \u00b7 150-char Overlap</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                with st.spinner("Running Strategy B..."):
                    try:
                        res_b = run_rag(cq, strategy="B", top_k=top_k, threshold=threshold)
                        render_answer(res_b)
                    except Exception as exc:
                        st.error(str(exc))
                        res_b = None

            if res_a and res_b:
                st.markdown("<hr>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">\U0001f4ca &nbsp;Comparison Metrics</div>',
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Score A", f"{res_a['top_score']:.3f}")
                with m2:
                    delta = res_b["top_score"] - res_a["top_score"]
                    st.metric("Score B", f"{res_b['top_score']:.3f}", delta=f"{delta:+.3f}")
                with m3:
                    st.metric("A Gated?", "Yes \u26d4" if res_a["gated"] else "No \u2705")
                with m4:
                    st.metric("B Gated?", "Yes \u26d4" if res_b["gated"] else "No \u2705")
                with m5:
                    if res_a["gated"] == res_b["gated"]:
                        winner = "Tied \U0001f91d"
                        banner_cls = "winner-tie"
                    elif not res_b["gated"] and res_a["gated"]:
                        winner = "B Wins \U0001f3c6"
                        banner_cls = "winner-b"
                    elif not res_a["gated"] and res_b["gated"]:
                        winner = "A Wins \U0001f3c6"
                        banner_cls = "winner-a"
                    else:
                        winner = "B Wins \U0001f3c6" if res_b["top_score"] > res_a["top_score"] else "A Wins \U0001f3c6"
                        banner_cls = "winner-b" if "B" in winner else "winner-a"
                    st.metric("Winner", winner)

                st.markdown(
                    f'<div class="winner-banner {banner_cls}">\U0001f3c6 &nbsp; {winner} &nbsp; \u2014 based on similarity score &amp; gate status</div>',
                    unsafe_allow_html=True,
                )



# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evaluation Results
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### 📊 Evaluation — 20 Test Questions")

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "evaluation", "results.json"
    )

    if not os.path.exists(results_path):
        st.info(
            "**Evaluation not yet run.**\n\n"
            "Execute the evaluator to generate results:\n\n"
            "```bash\npython evaluation/evaluator.py\n```\n\n"
            "Results will appear here automatically once the file is created."
        )
    else:
        import pandas as pd

        with open(results_path, encoding="utf-8") as fh:
            eval_results: list[dict] = json.load(fh)

        answerable   = [r for r in eval_results if     r["answerable"]]
        unanswerable = [r for r in eval_results if not r["answerable"]]

        # ── Summary metrics ────────────────────────────────────────────────
        st.markdown("#### Summary")

        def _pct_metric(lst: list, key: str) -> str:
            c = sum(1 for r in lst if r.get(key))
            return f"{c}/{len(lst)}" if lst else "N/A"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Answerable — Strategy A", _pct_metric(answerable, "correct_A"))
        with c2:
            st.metric("Answerable — Strategy B", _pct_metric(answerable, "correct_B"))
        with c3:
            st.metric("Unanswerable rejected — A", _pct_metric(unanswerable, "correct_A"))
        with c4:
            st.metric("Unanswerable rejected — B", _pct_metric(unanswerable, "correct_B"))

        # ── Full results table ─────────────────────────────────────────────
        st.markdown("#### All 20 Questions")
        df_rows = []
        for r in eval_results:
            df_rows.append(
                {
                    "#":         r["id"],
                    "Question":  r["question"],
                    "In Corpus": "✓" if r["answerable"] else "✗",
                    "Score A":   f"{r.get('top_score_A', 0):.3f}",
                    "Gated A":   "⛔" if r.get("gated_A") else "✅",
                    "Correct A": "✓" if r.get("correct_A") else "✗",
                    "Score B":   f"{r.get('top_score_B', 0):.3f}",
                    "Gated B":   "⛔" if r.get("gated_B") else "✅",
                    "Correct B": "✓" if r.get("correct_B") else "✗",
                }
            )
        df = pd.DataFrame(df_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Divergence examples ────────────────────────────────────────────
        divergences = [
            r
            for r in eval_results
            if r.get("gated_A") != r.get("gated_B")
            or r.get("correct_A") != r.get("correct_B")
        ]

        st.markdown(
            f"#### 🔍 Divergence Examples  "
            f"({len(divergences)} question(s) where strategies disagreed)"
        )

        if divergences:
            for r in divergences[:5]:
                with st.expander(f"Q{r['id']}: {r['question']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Strategy A**")
                        st.markdown(
                            f"Score: `{r.get('top_score_A', 0):.3f}` &nbsp;|&nbsp; "
                            f"Gated: `{'Yes' if r.get('gated_A') else 'No'}`"
                        )
                        st.markdown(r.get("answer_A") or "_No answer_")
                    with col2:
                        st.markdown("**Strategy B**")
                        st.markdown(
                            f"Score: `{r.get('top_score_B', 0):.3f}` &nbsp;|&nbsp; "
                            f"Gated: `{'Yes' if r.get('gated_B') else 'No'}`"
                        )
                        st.markdown(r.get("answer_B") or "_No answer_")
        else:
            st.success("Both strategies agreed on every question.")
