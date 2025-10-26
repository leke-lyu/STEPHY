# phyloGNN

Graph Neural Network (GNN) for learning migration patterns from pathogen phylogenies.

## Overview

This project views distinct populations as nodes within a connected graph. The framework integrates epidemic data as node features and converts phylogenetic trees into edge features, aiming to infer and predict pathogen migration dynamics between nodes through both node classification and link prediction tasks.

## Project Structure

- `mySimulation/` - Epidemic outbreak simulation pipeline
  - Sample SIR+M parameters and generate XML files
  - Simulate epidemic trajectories using the [ReMASTER](https://tgvaughan.github.io/remaster/) package from BEAST2
  - Extract node and edge features (node features from trajectory files and edge features from tree files) to prepare raw graph input data

## Status

Currently under active development. The GAT architecture is being implemented.

## Requirements

- Python 3.x
- R
- BEAST2 with the ReMASTER package

## Usage

Coming soon.
