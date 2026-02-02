# Multivariate Visual Analytics for Italian Regional Demographic and Healthcare Trends

**Course:** Visual Analytics 2025/2026 — Sapienza University of Rome  
**Authors:** A. Pagnotta, F. Trionfetti, M. Sorrentini

![Dashboard Screenshot](https://via.placeholder.com/1000x500?text=Dashboard+Overview+Placeholder)
*An integrated Single-Page Application for exploring high-dimensional demographic data.*

## 📋 Abstract
Public demographic and healthcare datasets (e.g., ISTAT) offer rich information but are difficult to explore due to high dimensionality. Existing platforms often force users into univariate analysis, making it hard to find complex correlations.

This project presents a **Visual Analytics dashboard** that enables the exploration of **32 indicators** across **20 Italian regions** over a 50-year span (2002–2050). By integrating dimensionality reduction (PCA) and clustering with coordinated geographic and temporal views, the system detects complex regional patterns.

## ✨ Key Features

* **Multivariate Analysis:** Projects 32 quantitative indicators into interpretable visual structures using PCA.
* **Visual Triggering Paradigm:** Eliminates complex menus. Algorithms like *Feature Differentiation* are activated via direct interaction (brushing/lasso) on the charts.
* **Spatiotemporal Exploration:** Navigates historical data and predictive models from **2002 to 2050**.
* **Predictive Modeling:** Includes future projections generated using a chained **ARIMA/ARIMAX** forecasting architecture.
* **Coordinated Multiple Views (CMV):** Selections propagate instantly across the Map, Bubble Chart, and PCA views using consistent highlighting.

## ⚙️ Installation and Setup

To run the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AntonioPagnotta/VA_project](https://github.com/AntonioPagnotta/VA_project)
    cd VA_project
    ```

2.  **Backend Setup:**
    Navigate to the backend directory and install the required Python dependencies:
    ```bash
    cd backend
    pip install -r requirements.txt
    python app.py
    ```

3.  **Frontend Setup:**
    Open `index.html` in a modern web browser. For the best experience (and to avoid local file CORS restrictions), it is recommended to serve the frontend using a local HTTP server (e.g., VS Code Live Server or Python's `http.server`).

## 📊 Dashboard Components

The interface follows the "Overview first, zoom and filter, then details-on-demand" mantra.

### 1. Context Views (Left Column)
* **Geospatial Distribution (Choropleth Map):** Built with **Leaflet.js** using NUTS-2 boundaries. It visualizes cluster membership or specific variable intensity (Heatmap mode).
* **Dimensionality Reduction (PCA):** A scatter plot projecting multivariate profiles onto 2D space. Supports **K-Means clustering** (Auto/Manual K) to group regions with similar profiles.

### 2. Detail Views (Right Column)
* **Bubble Chart (Gapminder-style):** A trivariate scatter plot (X, Y, Size) to investigate direct relationships. Features **historical trails** to visualize regional evolution over time.
* **Analytical Table (Feature Differentiation):** Automatically identifies specific variables that make a selected group distinct using a **Top-K Ranking** algorithm.

## 🛠️ Technology Stack

The system uses a modular Client-Server architecture to ensure responsiveness.

**Backend**
* **Language:** Python
* **Framework:** Flask (RESTful APIs)
* **Analytics:** Pandas (data manipulation), Scikit-learn (PCA & K-Means)

**Frontend**
* **Framework:** Single Page Application (SPA) with Vanilla JavaScript
* **Visualizations:** Chart.js (Scatter/Bubble), Leaflet.js (Map)
* **Design:** CSS Grid, Color Brewer scales for accessibility

## 🧮 Methodology

### The "Visual Triggering" Loop
To minimize cognitive load, analytics are triggered by geometry:
1.  **Action:** User brushes a lasso around points in the PCA plot.
2.  **Computation:** Server calculates the discriminativity score for all 32 variables.
3.  **Result:** The Analytical Table updates to show the Top-K features that distinguish the selection.

**Scoring Formula:**
$$Score = \frac{|\mu_{selected} - \mu_{rest}|}{\sigma_{global}}$$

## 🚀 Future Work
* Expansion to **NUTS-3 (Provincial)** granularity to reveal local disparities.
* Integration of **"What-If" simulations** for policy-making scenarios.

---
*Project Repository: [https://github.com/AntonioPagnotta/VA_project](https://github.com/AntonioPagnotta/VA_project)*