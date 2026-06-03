import { useEffect, useRef, useState } from "react";
import axios from "axios";
import "./App.css";

const API = "http://127.0.0.1:8000";
const THEME_KEY = "finrisk-theme";

const INIT = {
  age: 35,
  annual_income: 60000,
  credit_score: 650,
  employment_status: "Employed",
  loan_amount: 20000,
  loan_duration: 36,
  debt_to_income_ratio: 0.3,
  previous_loan_defaults: 0,
  bankruptcy_history: 0,
  savings_account_balance: 15000,
  total_assets: 80000,
  total_liabilities: 25000,
  net_worth: 55000,
  loan_purpose: "Business",
  question: "Should I take this loan for my business in Sri Lanka?",
};

const FIELDS = [
  { l: "Age", n: "age" },
  { l: "Annual Income (LKR)", n: "annual_income" },
  { l: "Credit Score", n: "credit_score" },
  { l: "Loan Amount (LKR)", n: "loan_amount" },
  { l: "Loan Duration (months)", n: "loan_duration" },
  { l: "Debt-to-Income Ratio", n: "debt_to_income_ratio" },
  { l: "Previous Defaults", n: "previous_loan_defaults" },
  { l: "Bankruptcy (0/1)", n: "bankruptcy_history" },
  { l: "Savings (LKR)", n: "savings_account_balance" },
  { l: "Total Assets (LKR)", n: "total_assets" },
  { l: "Total Liabilities (LKR)", n: "total_liabilities" },
  { l: "Net Worth (LKR)", n: "net_worth" },
];

const METRICS = [
  { v: "95.47%", l: "Model Recovery" },
  { v: "91%", l: "R2 Score" },
  { v: "20k+", l: "Training Records" },
  { v: "2 AI", l: "Models Ensembled" },
];

const rc = (score) => (score < 45 ? "low" : score < 60 ? "mid" : "high");
const rl = (score) =>
  score < 45 ? "LOW RISK" : score < 60 ? "MEDIUM RISK" : "HIGH RISK";

function getInitialTheme() {
  const saved = window.localStorage.getItem(THEME_KEY);
  if (saved === "dark" || saved === "light") return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function App() {
  const [form, setForm] = useState(INIT);
  const [result, setResult] = useState(null);
  const [loading, setLoad] = useState(false);
  const [error, setErr] = useState(null);
  const [tab, setTab] = useState("analyzer");
  const [csv, setCsv] = useState(null);
  const [csvR, setCsvR] = useState(null);
  const [fu, setFu] = useState("");
  const [fuR, setFuR] = useState(null);
  const [fuLoad, setFuLoad] = useState(false);
  const [theme, setTheme] = useState(getInitialTheme);
  const fileRef = useRef(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem(THEME_KEY, theme);
  }, [theme]);

  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((f) => ({ ...f, [name]: isNaN(value) || value === "" ? value : Number(value) }));
  };

  const onPredict = async () => {
    setLoad(true);
    setErr(null);
    setResult(null);
    setFuR(null);
    try {
      const r = await axios.post(`${API}/predict`, form);
      setResult(r.data);
    } catch {
      setErr("Cannot reach backend. Make sure uvicorn is running on port 8000.");
    } finally {
      setLoad(false);
    }
  };

  const onFollowUp = async () => {
    if (!fu.trim()) return;
    setFuLoad(true);
    try {
      const r = await axios.post(`${API}/chat`, {
        message: fu,
        risk_score: result?.risk_score,
      });
      setFuR(r.data.response);
      setFu("");
    } catch {
      setFuR("Could not get response. Please try again.");
    } finally {
      setFuLoad(false);
    }
  };

  const onUpload = async () => {
    if (!csv) return;
    const fd = new FormData();
    fd.append("file", csv);
    try {
      const r = await axios.post(`${API}/upload-csv`, fd);
      setCsvR(r.data);
    } catch {
      setErr("CSV upload failed.");
    }
  };

  return (
    <div className="app-shell">
      <div className="bg-wrap" aria-hidden="true">
        <div className="bg-base" />
        <div className="bg-orb bg-orb-a" />
        <div className="bg-orb bg-orb-b" />
        <div className="bg-orb bg-orb-c" />
        <div className="bg-shape bg-shape-a" />
        <div className="bg-shape bg-shape-b" />
        <div className="bg-grid" />
      </div>

      <div className="app">
        <nav className="nav">
          <div className="nav-i">
            <div className="nav-brand">
              <div className="nav-icon">F</div>
              <div>
                <div className="nav-name">FinRisk AI</div>
                <div className="nav-sub">Fast intelligence for Sri Lanka</div>
              </div>
            </div>

            <div className="nav-actions">
              <div className="nav-pills">
                <div className="npill model">XGBoost · LightGBM · Gemini</div>
                <div className="npill live">
                  <span className="live-dot" />
                  API Live
                </div>
              </div>

              <div className="theme-switch" role="group" aria-label="Theme switch">
                <button
                  className={`theme-btn ${theme === "light" ? "on" : ""}`}
                  onClick={() => setTheme("light")}
                  type="button"
                >
                  Light
                </button>
                <button
                  className={`theme-btn ${theme === "dark" ? "on" : ""}`}
                  onClick={() => setTheme("dark")}
                  type="button"
                >
                  Dark
                </button>
              </div>
            </div>
          </div>
        </nav>

        <header className="hero">
          <div className="hero-badge">Built for Sri Lanka's Financial Landscape</div>
          <h1 className="hero-title au d1">
            Smarter financial
            <br />
            <span className="grad">risk decisions</span>
          </h1>
          <p className="hero-desc au d2">
            Instant risk scores, loan decisions, and Gemini-powered advice,
            calibrated for Sri Lanka's economic reality.
          </p>

          <div className="hero-stats">
            {METRICS.map((s, i) => (
              <div className={`hstat au d${i + 3}`} key={s.l}>
                <div className="hstat-v">{s.v}</div>
                <div className="hstat-l">{s.l}</div>
              </div>
            ))}
          </div>
        </header>

        <main className="wrap">
          <div className="tabs-row au d7">
            <div className="tabs-inner">
              {[
                { id: "analyzer", l: "Risk Analyzer" },
                { id: "upload", l: "Upload CSV" },
              ].map((t) => (
                <button
                  key={t.id}
                  onClick={() => setTab(t.id)}
                  className={`tab-btn ${tab === t.id ? "on" : ""}`}
                  type="button"
                >
                  {t.l}
                </button>
              ))}
            </div>
          </div>

          {tab === "analyzer" && (
            <div className="two-col">
              <section className="card asr">
                <div className="card-kicker">Financial Profile</div>
                <h2 className="card-title">Build your risk case</h2>
                <p className="card-desc">
                  Enter the key indicators and let the model score and Gemini advise.
                </p>

                <label className="flabel" htmlFor="question" style={{ marginBottom: 7 }}>
                  Your question to the AI
                </label>
                <textarea
                  id="question"
                  name="question"
                  value={form.question}
                  onChange={onChange}
                  className="finput q-inp"
                  rows={3}
                  style={{ marginBottom: 18 }}
                  placeholder="e.g. Should I open a shop in Colombo?"
                />

                <div className="fgrid">
                  {FIELDS.map(({ l, n }) => (
                    <div className="fgroup" key={n}>
                      <label className="flabel" htmlFor={n}>
                        {l}
                      </label>
                      <input
                        id={n}
                        type="number"
                        name={n}
                        value={form[n]}
                        onChange={onChange}
                        className="finput"
                      />
                    </div>
                  ))}
                  <div className="fgroup">
                    <label className="flabel" htmlFor="employment_status">
                      Employment Status
                    </label>
                    <select
                      id="employment_status"
                      name="employment_status"
                      value={form.employment_status}
                      onChange={onChange}
                      className="finput"
                    >
                      {["Employed", "Unemployed", "Self-Employed"].map((o) => (
                        <option key={o}>{o}</option>
                      ))}
                    </select>
                  </div>
                  <div className="fgroup">
                    <label className="flabel" htmlFor="loan_purpose">
                      Loan Purpose
                    </label>
                    <select
                      id="loan_purpose"
                      name="loan_purpose"
                      value={form.loan_purpose}
                      onChange={onChange}
                      className="finput"
                    >
                      {["Business", "Personal", "Education", "Home"].map((o) => (
                        <option key={o}>{o}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <button onClick={onPredict} disabled={loading} className="btn-main" type="button">
                  <span className="btn-inner">
                    {loading ? (
                      <>
                        <svg
                          className="spin-icon"
                          viewBox="0 0 24 24"
                          fill="none"
                          aria-hidden="true"
                        >
                          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" opacity="0.25" />
                          <path fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                        </svg>
                        Analyzing with AI...
                      </>
                    ) : (
                      <>
                        Analyze My Risk
                        <span className="btn-arrow">→</span>
                      </>
                    )}
                  </span>
                </button>

                {error && <div className="err-box">{error}</div>}
              </section>

              <section className="card asl">
                <div className="card-kicker">Results</div>
                <h2 className="card-title">Your risk report</h2>
                <p className="card-desc">Score, loan decision, financial chart, and Gemini advice.</p>

                {!result && !loading && (
                  <div className="empty">
                    <div className="empty-ico">AI</div>
                    <h3>Ready when you are</h3>
                    <p>Complete the profile on the left and click Analyze to generate your report.</p>
                  </div>
                )}

                {loading && (
                  <div className="load-state">
                    <div className="spin" />
                    <h3 className="load-title">Analyzing...</h3>
                    <p className="load-sub">Running ML models and calling Gemini.</p>
                  </div>
                )}

                {result && (
                  <div className="result-stack">
                    <div className={`score-card ${rc(result.risk_score)}`}>
                      <div className="score-left">
                        <div className="score-label">RISK SCORE</div>
                        <div className="score-num">{result.risk_score}</div>
                        <div className="score-risk">{rl(result.risk_score)}</div>
                      </div>
                      <div className="score-right">
                        <div className={`dec-chip ${result.loan_approved === 1 ? "ok" : "no"}`}>
                          {result.loan_approved === 1 ? "Approved" : "Review"} {result.loan_decision}
                        </div>
                        <div className="pw-by">XGBoost · LightGBM</div>
                      </div>
                    </div>

                    {result.chart && (
                      <div className="chart-wrap">
                        <img src={`data:image/png;base64,${result.chart}`} alt="Risk chart" />
                      </div>
                    )}

                    <div className="ai-box">
                      <div className="ai-head">
                        <div className="ai-avatar">AI</div>
                        <div>
                          <div className="ai-name">Gemini AI Advice</div>
                          <div className="ai-sub">Personalized from your financial profile</div>
                        </div>
                      </div>
                      <p className="ai-text">{result.gemini_response}</p>
                    </div>

                    <div className="lk-note">
                      <span className="lk-flag">LK</span>
                      <span className="lk-text">{result.sri_lanka_note}</span>
                    </div>

                    <div className="chat-area">
                      <div className="chat-title">ASK A FOLLOW UP QUESTION</div>
                      <div className="chat-row">
                        <input
                          type="text"
                          value={fu}
                          onChange={(e) => setFu(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && onFollowUp()}
                          placeholder="Eg. Should I wait 6 months before investing?"
                          className="finput"
                          style={{ flex: 1, marginBottom: 0 }}
                        />
                        <button onClick={onFollowUp} disabled={fuLoad} className="btn-send" type="button">
                          {fuLoad ? "Sending..." : "Send"}
                        </button>
                      </div>
                      {fuR && <div className="chat-resp">{fuR}</div>}
                    </div>
                  </div>
                )}
              </section>
            </div>
          )}

          {tab === "upload" && (
            <div className="upload-layout">
              <section className="card au upload-card">
                <div className="card-kicker">Bulk Analysis</div>
                <h2 className="card-title">Upload a CSV dataset</h2>
                <p className="card-desc">Preview your data structure before running deeper analysis.</p>

                <div className="upload-zone" onClick={() => fileRef.current?.click()} role="button" tabIndex={0}>
                  <div className="up-ico">CSV</div>
                  <div className="up-title">{csv ? csv.name : "Drop or click to choose a CSV file"}</div>
                  <div className="up-sub">Supports .csv files of any size</div>
                  <input
                    ref={fileRef}
                    type="file"
                    accept=".csv"
                    style={{ display: "none" }}
                    onChange={(e) => setCsv(e.target.files[0])}
                  />
                </div>

                <button onClick={onUpload} disabled={!csv} className="btn-upload" type="button">
                  Upload & Analyze
                </button>

                {csvR && (
                  <div className="upload-results au">
                    <div className="csv-grid">
                      <div className="csv-stat g">
                        <div className="csv-stat-v">{csvR.rows}</div>
                        <div className="csv-stat-l">Total Rows</div>
                      </div>
                      <div className="csv-stat b">
                        <div className="csv-stat-v">{csvR.columns.length}</div>
                        <div className="csv-stat-l">Columns</div>
                      </div>
                    </div>

                    <div className="mini-card">
                      <div className="flabel">Column names</div>
                      <div className="tag-list">
                        {csvR.columns.map((c, i) => (
                          <span key={c} className="tag" style={{ animationDelay: `${i * 0.03}s` }}>
                            {c}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="mini-card">
                      <div className="flabel">Data preview</div>
                      <pre className="pre-wrap">{JSON.stringify(csvR.preview, null, 2)}</pre>
                    </div>
                  </div>
                )}
              </section>
            </div>
          )}
        </main>

        <footer className="footer">FinRisk AI · XGBoost · LightGBM · Gemini AI · Built for Sri Lanka</footer>
      </div>
    </div>
  );
}
