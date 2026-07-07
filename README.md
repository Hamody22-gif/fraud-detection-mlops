# Fraud Detection — MLOps Pipeline

An end-to-end machine learning system that detects fraudulent credit-card
transactions, built to production standards: experiment tracking, a
reproducible data pipeline, orchestrated retraining, real-time serving, and
drift monitoring.

> 🚧 **Building in public**, one phase at a time.

## Dataset
Sparkov simulated credit-card transactions (`fraudTrain.csv` / `fraudTest.csv`) —
a highly imbalanced dataset with raw transaction features (merchant, category,
amount, time, location), requiring real feature engineering.

## Tech stack (planned)
Python · uv · ruff · mypy · pytest · MLflow · DVC · Airflow · Feast · Redis ·
FastAPI · Docker · Kafka · Prometheus · Grafana · Evidently AI · GitHub Actions ·
Terraform

## Roadmap
- [x] **Phase 0** — Foundations: repo, environment, quality tooling
- [ ] **Phase 1** — Data exploration + baseline model (LogReg, XGBoost)
- [ ] **Phase 2** — Experiment tracking + registry (MLflow)
- [ ] **Phase 3** — Reproducible pipeline (DVC)
- [ ] **Phase 4** — Orchestration (Airflow)
- [ ] **Phase 5** — Feature store (Feast)
- [ ] **Phase 6** — Serving API (FastAPI + Docker) ← MVP
- [ ] **Phase 7** — Streaming (Kafka)
- [ ] **Phase 8** — Monitoring + drift (Prometheus, Grafana, Evidently)
- [ ] **Phase 9** — CI/CD (GitHub Actions)
- [ ] **Phase 10** — Infra as Code + cloud deploy (Terraform)
- [ ] **Phase 11** — A/B testing + model routing
- [ ] **Phase 12** — Polish & write-up
