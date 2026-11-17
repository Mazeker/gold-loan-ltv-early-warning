PY ?= python

.PHONY: all data experiments agent test clean

all: experiments agent

data:            ## refresh raw data from Yahoo Finance (optional; snapshot is included)
	$(PY) scripts/fetch_data.py

experiments:     ## models, loan-book backtest, tables and figures (about 3 minutes on 4 cores)
	cd src && $(PY) -m goldltv.pipeline

agent:           ## today's agent run and the monthly replay used in the paper
	cd src && $(PY) -m goldltv.agent
	$(PY) scripts/evaluate_agent.py

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf results/figures/* results/tables/* data/processed/* agent_outbox/pending/*
