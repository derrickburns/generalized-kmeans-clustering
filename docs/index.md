# Generalized K-Means Clustering

A Scala library for generalized K-means clustering using Bregman divergences on Apache Spark.

## Overview

This library implements generalized K-means clustering algorithms that work with various distance functions through Bregman divergences, extending beyond traditional Euclidean distance clustering.

## Features

- Multiple Bregman divergence implementations (Euclidean, Kullback-Leibler, Itakura-Saito, etc.)
- Scalable clustering on Apache Spark
- Weighted clustering support
- Multiple initialization strategies (K-means++, random)
- Mini-batch and column-tracking optimizations

## Getting Started

See the [README](../README.md) for detailed usage instructions and examples.

## API Documentation

For detailed API documentation, please refer to the Scaladoc in the source code.