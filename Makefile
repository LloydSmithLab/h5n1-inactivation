# make preamble via https://tech.davis-hansson.com/p/make/
SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX := >
.DEFAULT_GOAL := all

######################
# shell configuration
#####################
MKDIR := @mkdir -p
RM := @rm
RMDIR := @rmdir
PYTHON := python3
ECHO := @echo

##############################
# directory and file structure
##############################
SRC := src
DAT := dat
OUT := out

RAW := $(DAT)/raw
CLEANED := $(DAT)/cleaned
PRIOR_CONFIG := $(DAT)/prior_config

CHAINS := $(OUT)/chains
FIGURES := $(OUT)/figures
TABLES := $(OUT)/tables
DIAGNOSTICS := $(TABLES)/diagnostics


TITER_PRIORS := $(shell find $(PRIOR_CONFIG) -name \
   priors_individual_titer_*.toml)
HALFLIFE_PRIORS := $(shell find $(PRIOR_CONFIG) -name \
   priors_halflife_*.toml)
ALL_PRIORS := $(TITER_PRIORS) $(HALFLIFE_PRIORS)

TITER_MODEL_NAMES := $(patsubst \
   $(PRIOR_CONFIG)/priors_individual_titer_%.toml, %, $(TITER_PRIORS))
HALFLIFE_MODEL_NAMES := $(patsubst \
   $(PRIOR_CONFIG)/priors_halflife_%.toml, %, $(HALFLIFE_PRIORS))
ALL_MODEL_NAMES := $(TITER_MODEL_NAMES) $(HALFLIFE_MODEL_NAMES)

MCMC_CONFIG := $(DAT)/mcmc_config.toml
ALL_CONFIGS := $(ALL_PRIORS) $(MCMC_CONFIG)

RAW_DATA_63C := $(RAW)/data_63C.xlsx
RAW_DATA_72C := $(RAW)/data_72C.xlsx
CLEANED_DATA := $(CLEANED)/data.tsv


DEFAULT_CHAIN_DEPS := $(CLEANED_DATA) $(MCMC_CONFIG)
DEFAULT_TITER_CHAINS = $(CHAINS)/individual_titer_default.pickle
DEFAULT_HALFLIFE_CHAINS = $(CHAINS)/halflife_default.pickle
ALL_TITER_CHAINS := $(patsubst %, $(CHAINS)/individual_titer_%.pickle, \
   $(TITER_MODEL_NAMES))
ALL_HALFLIFE_CHAINS := $(patsubst %, $(CHAINS)/halflife_%.pickle, \
   $(HALFLIFE_MODEL_NAMES))
ALL_CHAINS := $(ALL_TITER_CHAINS) $(ALL_HALFLIFE_CHAINS)

DEFAULT_FIGURE_DEPS := $(CLEANED_DATA) $(DEFAULT_TITER_CHAINS) \
   $(DEFAULT_HALFLIFE_CHAINS)
FIT_FIGURES := $(patsubst %, $(FIGURES)/figure-fit-%.pdf, \
   $(HALFLIFE_MODEL_NAMES))
PRIOR_CHECK_FIGURES := $(patsubst %, \
   $(FIGURES)/figure-prior-check-%.pdf, \
   $(HALFLIFE_MODEL_NAMES))

ALL_FIGURES := $(FIT_FIGURES) $(PRIOR_CHECK_FIGURES)

TABLE_TITERS := $(TABLES)/titers.tsv
HALFLIFE_TABLES := $(patsubst %, $(TABLES)/table_halflives_%.tsv, \
   $(HALFLIFE_MODEL_NAMES))
SENSITIVITY_TABLES := $(TABLES)/table_halflife_prior_sensitivity.tsv

DIAGNOSTICS_RAW := $(patsubst $(CHAINS)/%.pickle, \
   $(DIAGNOSTICS)/%_mcmc_diagnostics.tsv, $(ALL_CHAINS))
DIAGNOSTICS_SUMMARY := $(subst .tsv,_extrema.tsv, \
   $(DIAGNOSTICS_RAW))
DIAGNOSTIC_TABLES = $(DIAGNOSTICS_RAW) $(DIAGNOSTICS_SUMMARY)
ALL_TABLES = $(TABLE_TITERS) $(HALFLIFE_TABLES) $(DIAGNOSTIC_TABLES) \
   $(SENSITIVITY_TABLES)

DEFAULT_TABLE_DEPS := $(DEFAULT_FIGURE_DEPS)

########
# Rules
########
$(CLEANED_DATA): $(SRC)/clean_data.py $(RAW_DATA_63C) $(RAW_DATA_72C)
> $(MKDIR) $(CLEANED)
> $(PYTHON) $^ $@


$(CHAINS)/individual_titer_%.pickle: $(SRC)/fit_model.py $(DEFAULT_CHAIN_DEPS) $(PRIOR_CONFIG)/priors_individual_titer_%.toml
> $(MKDIR) $(CHAINS)
> $(PYTHON) $^ individual_titer -o $@

$(CHAINS)/halflife_%.pickle: $(SRC)/fit_model.py $(DEFAULT_CHAIN_DEPS) \
   $(PRIOR_CONFIG)/priors_halflife_%.toml
> $(MKDIR) $(CHAINS)
> $(PYTHON) $^ halflife -o $@

$(FIGURES)/figure-fit-%.pdf: $(SRC)/figure_fit.py $(CLEANED_DATA) \
   $(DEFAULT_TITER_CHAINS) $(CHAINS)/halflife_%.pickle
> $(MKDIR) $(FIGURES)
> $(PYTHON) $^ $@

$(FIGURES)/figure-prior-check-%.pdf: $(SRC)/figure_prior_check.py \
   $(CLEANED_DATA) $(DEFAULT_TITER_CHAINS) $(CHAINS)/halflife_%.pickle
> $(MKDIR) $(FIGURES)
> $(PYTHON) $^ $@

$(TABLE_TITERS): $(SRC)/table_titers.py $(DEFAULT_TABLE_DEPS)
> $(MKDIR) $(TABLES)
> $(PYTHON) $^ $@

$(TABLES)/table_halflives_%.tsv: $(SRC)/table_halflives.py \
  $(CLEANED_DATA) $(DEFAULT_TITER_CHAINS) $(CHAINS)/halflife_%.pickle
> $(MKDIR) $(TABLES)
> $(PYTHON) $^ $@

$(TABLES)/table_halflife_prior_sensitivity.tsv: \
  $(SRC)/table_halflife_prior_sensitivity.py $(HALFLIFE_TABLES)
> $(MKDIR) $(TABLES)
> $(PYTHON) $^ -o $@

$(DIAGNOSTICS)/individual_titer_%_mcmc_diagnostics.tsv: \
   $(SRC)/table_diagnostics.py \
   $(CHAINS)/individual_titer_%.pickle
> $(MKDIR) $(DIAGNOSTICS)
> $(PYTHON) $^ $@

$(DIAGNOSTICS)/halflife_%_mcmc_diagnostics.tsv: \
   $(SRC)/table_diagnostics.py \
   $(CHAINS)/halflife_%.pickle
> $(MKDIR) $(DIAGNOSTICS)
> $(PYTHON) $^ $@

$(DIAGNOSTICS)/%_extrema.tsv: $(SRC)/table_diagnostic_extrema.py \
  $(DIAGNOSTICS)/%.tsv
> $(MKDIR) $(DIAGNOSTICS)
> $(PYTHON) $^ $@


ALL_TARGETS := $(CLEANED_DATA) $(ALL_CHAINS) $(ALL_FIGURES) $(ALL_TABLES)
all: $(ALL_TARGETS)

##########################
# Phony rules / shortcuts
##########################

.PHONY: clean deltemp list_models list_configs list_chains \
list_figures list_tables list_targets

# delete emacs tempfiles
deltemp:
> $(RM) -f ./**/*~
> $(RM) -f ./**/*#
> $(RM) -f ./**/.DS_Store

clean: deltemp
> $(RM) -f $(SRC)/__pycache__/*
> $(RM) -f $(ALL_TARGETS)
> $(MKDIR) $(CHAINS) $(FIGURES) $(DIAGNOSTICS) \
   $(TABLES) $(OUT) $(CLEANED) \
   $(SRC)/__pycache__
> $(RMDIR) $(CHAINS) $(FIGURES) $(DIAGNOSTICS) \
  $(TABLES) $(OUT) $(CLEANED) \
  $(SRC)/__pycache__

list_models:
> $(ECHO) $(ALL_MODEL_NAMES)

list_configs:
> $(ECHO) $(ALL_CONFIGS)

list_chains:
> $(ECHO) $(ALL_CHAINS)

list_figures:
> $(ECHO) $(ALL_FIGURES)

list_tables:
> $(ECHO) $(ALL_TABLES)

list_targets:
> $(ECHO) $(ALL_TARGETS)
