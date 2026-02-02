# Multivariate Visual Analytics for Italian Regional Demographic and Healthcare Trends

**Course:** Visual Analytics 2025/2026 — Sapienza University of Rome  
[cite_start]**Authors:** A. Pagnotta, F. Trionfetti, M. Sorrentini [cite: 3]

![Dashboard Screenshot](https://via.placeholder.com/1000x500?text=Dashboard+Overview+Placeholder)
*An integrated Single-Page Application for exploring high-dimensional demographic data.*

## 📋 Abstract
Public demographic and healthcare datasets (e.g., ISTAT) offer rich information but are difficult to explore due to high dimensionality. [cite_start]Existing platforms often force users into univariate analysis, making it hard to find complex correlations[cite: 5, 6].

[cite_start]This project presents a **Visual Analytics dashboard** that enables the exploration of **32 indicators** across **20 Italian regions** over a 50-year span (2002–2050)[cite: 7]. [cite_start]By integrating dimensionality reduction (PCA) and clustering with coordinated geographic and temporal views, the system detects complex regional patterns[cite: 8].

## ✨ Key Features

* [cite_start]**Multivariate Analysis:** Projects 32 quantitative indicators into interpretable visual structures using PCA[cite: 75].
* **Visual Triggering Paradigm:** Eliminates complex menus. [cite_start]Algorithms like *Feature Differentiation* are activated via direct interaction (brushing/lasso) on the charts[cite: 9].
* [cite_start]**Spatiotemporal Exploration:** Navigates historical data and predictive models from **2002 to 2050**[cite: 36].
* [cite_start]**Predictive Modeling:** Includes future projections generated using a chained **ARIMA/ARIMAX** forecasting architecture[cite: 44].
* [cite_start]**Coordinated Multiple Views (CMV):** Selections propagate instantly across the Map, Bubble Chart, and PCA views using consistent highlighting[cite: 107].

## 📊 Dashboard Components

[cite_start]The interface follows the "Overview first, zoom and filter, then details-on-demand" mantra[cite: 51].

### 1. Context Views (Left Column)
* **Geospatial Distribution (Choropleth Map):** Built with **Leaflet.js** using NUTS-2 boundaries. [cite_start]It visualizes cluster membership or specific variable intensity (Heatmap mode)[cite: 61, 62].
* **Dimensionality Reduction (PCA):** A scatter plot projecting multivariate profiles onto 2D space. [cite_start]Supports **K-Means clustering** (Auto/Manual K) to group regions with similar profiles[cite: 75, 78].

### 2. Detail Views (Right Column)
* **Bubble Chart (Gapminder-style):** A trivariate scatter plot (X, Y, Size) to investigate direct relationships. [cite_start]Features **historical trails** to visualize regional evolution over time[cite: 86, 90].
* [cite_start]**Analytical Table (Feature Differentiation):** Automatically identifies specific variables that make a selected group distinct using a **Top-K Ranking** algorithm[cite: 96, 98].

## 🛠️ Technology Stack

[cite_start]The system uses a modular Client-Server architecture to ensure responsiveness[cite: 125].

**Backend**
* [cite_start]**Language:** Python [cite: 127]
* [cite_start]**Framework:** Flask (RESTful APIs) [cite: 128]
* [cite_start]**Analytics:** Pandas (data manipulation), Scikit-learn (PCA & K-Means) [cite: 127]

**Frontend**
* [cite_start]**Framework:** Single Page Application (SPA) with Vanilla JavaScript [cite: 128]
* [cite_start]**Visualizations:** Chart.js (Scatter/Bubble), Leaflet.js (Map) [cite: 129]
* [cite_start]**Design:** CSS Grid, Color Brewer scales for accessibility [cite: 52, 55]

## 🧮 Methodology

### The "Visual Triggering" Loop
[cite_start]To minimize cognitive load, analytics are triggered by geometry[cite: 114]:
1.  **Action:** User brushes a lasso around points in the PCA plot.
2.  **Computation:** Server calculates the discriminativity score for all 32 variables.
3.  **Result:** The Analytical Table updates to show the Top-K features that distinguish the selection.

**Scoring Formula:**
$$Score = \frac{|\mu_{selected} - \mu_{rest}|}{\sigma_{global}}$$
[cite_start][cite: 100]

## 🚀 Future Work
* [cite_start]Expansion to **NUTS-3 (Provincial)** granularity to reveal local disparities[cite: 181].
* [cite_start]Integration of **"What-If" simulations** for policy-making scenarios[cite: 182].

---
[cite_start]*Project Repository: [https://github.com/AntonioPagnotta/VA_project](https://github.com/AntonioPagnotta/VA_project)* [cite: 11]