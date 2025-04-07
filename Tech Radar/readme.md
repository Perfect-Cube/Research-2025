deployed link - https://tech-radar-bf5g.onrender.com/

AWS  -   http://13.235.0.185:5000/

```
While using TechRadarVisualizatio_Final update the Header.tsx and RadarVisualizatio.tsx files.
```

self service portal --> npm

```
cd TechRadarVisualizer
```
```
npm install
```
```
npm run dev
```


To make the local Python-only version of the Apptension Tech Radar UI as close as possible to the original, we need to replicate its key visual and functional elements within the constraints of Python libraries. The original Tech Radar, built with React and D3.js, features a circular radar plot with four quadrants (Techniques, Tools, Platforms, Languages) and four rings (Hold, Assess, Trial, Adopt), where technologies are plotted as points with labels, colored by ring status, and linked to detailed views. It also includes interactivity like hover effects and a clean, modern design.
Using Flask for the web server and Plotly for visualization, we’ll enhance the previous solution to better mimic these aspects:

    Quadrants and Rings: Position technologies within quadrants using angular offsets and spread them across rings with radial distances, matching the original layout.
    Colors and Styling: Use the exact color scheme from the source code (e.g., radar.tsx).
    Interactivity: Add hover tooltips with Plotly and clickable list items for detail pages.
    Data: Use a local JSON file structured like the original’s RadarTechnology interface.

Here’s the improved implementation:
Setup and Requirements

  Install Libraries:
  
        Run pip install Flask plotly jinja2 in your terminal.
    
  Prepare Local Data:
        Create data.json based on the RadarTechnology interface from radar.types.ts:
        json

        [
            {
                "id": "1",
                "label": "Python",
                "quadrant": "Languages",
                "ring": "Adopt",
                "description": "Versatile language for web and data science.",
                "isNew": true
            },
            {
                "id": "2",
                "label": "Flask",
                "quadrant": "Tools",
                "ring": "Trial",
                "description": "Lightweight Python web framework.",
                "isNew": false
            },
            {
                "id": "3",
                "label": "Docker",
                "quadrant": "Platforms",
                "ring": "Assess",
                "description": "Containerization platform.",
                "isNew": false
            },
            {
                "id": "4",
                "label": "TDD",
                "quadrant": "Techniques",
                "ring": "Hold",
                "description": "Test-driven development methodology.",
                "isNew": false
            }
        ]

  Save in your project directory.
    Project Structure:

    tech_radar_python/
    ├── app.py
    ├── data.json
    └── templates/
        ├── index.html
        └── technology_detail.html

Python Code
Create app.py with enhanced Plotly visualization and Flask routing:
python
```
from flask import Flask, render_template
import json
import plotly.graph_objects as go
import random

app = Flask(__name__)

# Load local data
with open("data.json", "r") as f:
    technologies = json.load(f)

# Map rings to radial values (adjusted for visual spread, closer to original)
ring_values = {"Hold": 1, "Assess": 3, "Trial": 5, "Adopt": 7}

# Map quadrants to theta ranges (0° to 360°, rotated to match original)
quadrant_ranges = {
    "Techniques": (0, 90),    # Top-right
    "Tools": (90, 180),       # Bottom-right
    "Platforms": (180, 270),  # Bottom-left
    "Languages": (270, 360)   # Top-left
}

# Colors from radar.tsx
color_map = {
    "Hold": "#F44336",   # Red
    "Assess": "#FFC107", # Yellow
    "Trial": "#FF9800",  # Orange
    "Adopt": "#4CAF50"   # Green
}

@app.route('/')
def index():
    # Group technologies by quadrant for plotting
    quadrant_groups = {}
    for tech in technologies:
        quadrant = tech["quadrant"]
        if quadrant not in quadrant_groups:
            quadrant_groups[quadrant] = []
        quadrant_groups[quadrant].append(tech)

    # Create radar chart
    fig = go.Figure()
    for quadrant, techs in quadrant_groups.items():
        min_theta, max_theta = quadrant_ranges[quadrant]
        theta_step = (max_theta - min_theta) / max(len(techs), 1)  # Spread within quadrant
        for i, tech in enumerate(techs):
            # Calculate theta with slight randomization for spread
            base_theta = min_theta + i * theta_step
            theta = base_theta + random.uniform(-theta_step / 4, theta_step / 4)
            r = ring_values.get(tech["ring"], 1)
            hover_text = f"{tech['label']}<br>{tech['ring']}<br>{tech['description']}"
            fig.add_trace(go.Scatterpolar(
                r=[r],
                theta=[theta],
                mode='markers+text',
                text=[tech["label"] if i % 2 == 0 else ""],  # Stagger labels to avoid overlap
                textposition="top center",
                marker=dict(size=12, color=color_map.get(tech["ring"], "#000")),
                hoverinfo="text",
                hovertext=[hover_text],
                name=tech["label"]
            ))

    # Customize layout to match Apptension Tech Radar
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 8],
                tickvals=[1, 3, 5, 7],
                ticktext=["Hold", "Assess", "Trial", "Adopt"],
                tickangle=0
            ),
            angularaxis=dict(
                rotation=-90,  # Rotate so Languages is top-left
                direction="clockwise",
                tickvals=[0, 90, 180, 270],
                ticktext=["Techniques", "Tools", "Platforms", "Languages"],
                tickfont=dict(size=14)
            ),
            bgcolor="#F5F5F5"  # Light gray background
        ),
        showlegend=False,
        title=dict(text="Tech Radar", x=0.5, font=dict(size=20)),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    plot_div = fig.to_html(full_html=False)
    return render_template("index.html", plot_div=plot_div, technologies=technologies)

@app.route('/technology/<tech_id>')
def technology_detail(tech_id):
    tech = next((t for t in technologies if t["id"] == tech_id), None)
    if tech:
        return render_template("technology_detail.html", tech=tech, color_map=color_map)
    return "Technology not found", 404

if __name__ == '__main__':
    app.run(debug=True)
```
HTML Templates

  Main Page (templates/index.html):
    html

    <!DOCTYPE html>
    <html>
    <head>
        <title>Tech Radar</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #F5F5F5; margin: 20px; }
            h1 { text-align: center; color: #333; }
            h2 { color: #666; }
            ul { list-style: none; padding: 0; }
            li { margin: 10px 0; }
            a { text-decoration: none; color: #007BFF; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>Tech Radar</h1>
        <div>{{ plot_div | safe }}</div>
        <h2>Technologies</h2>
        <ul>
            {% for tech in technologies %}
            <li>
                <a href="/technology/{{ tech.id }}">{{ tech.label }}</a>
                ({{ tech.quadrant }}, {{ tech.ring }}{% if tech.isNew %}, New{% endif %})
            </li>
            {% endfor %}
        </ul>
    </body>
    </html>

  Detail Page (templates/technology_detail.html):
    html

    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ tech.label }} - Tech Radar</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #F5F5F5; margin: 20px; }
            h1 { color: {{ color_map[tech.ring] }}; }
            p { color: #333; }
            a { color: #007BFF; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>{{ tech.label }}</h1>
        <p><strong>Quadrant:</strong> {{ tech.quadrant }}</p>
        <p><strong>Ring:</strong> {{ tech.ring }}</p>
        {% if tech.isNew %}<p><strong>Status:</strong> New</p>{% endif %}
        <p><strong>Description:</strong> {{ tech.description }}</p>
        <a href="/">Back to Radar</a>
    </body>
    </html>

Running the UI

    Save all files in tech_radar_python/.
    Open a terminal, navigate to the folder, and run:

    python app.py

    Open your browser to http://127.0.0.1:5000. You’ll see:
        A radar chart with technologies plotted in their quadrants and rings, colored by status.
        Labels staggered to reduce overlap, with hover tooltips showing details.
        A list below linking to detail pages, styled to match the original’s clean look.

How It Matches the Original

    Layout: Quadrants are positioned like the original (Techniques top-right, Tools bottom-right, etc.), with rotation=-90 to align Languages top-left, as seen in radar.tsx.
    Rings: Radial distances (1, 3, 5, 7) and labels (Hold, Assess, Trial, Adopt) match the concentric ring structure, adjusted for visual spread.
    Colors: Exact hex codes from radar.tsx (e.g., #F44336 for Hold) ensure visual consistency.
    Interactivity: Hover tooltips mimic the original’s hover effects, and clickable list items replicate navigation to details.
    Styling: Light gray background (#F5F5F5) and Arial font approximate the modern design.

Limitations

    Precision: Plotly’s scatterpolar doesn’t sort technologies within quadrants as precisely as D3.js, so random offsets are used to spread them, which may not perfectly match the original’s layout.
    Features: Missing advanced features like search, filtering, or dynamic quadrant lines due to Plotly’s limitations and Python-only constraint.
    Complexity: Adding more fields (e.g., alternatives, projects) from RadarTechnology would require expanding the detail page, but this keeps it simple yet close.

This is as close as we can get with Python only, leveraging Flask and Plotly to replicate the core experience of the Apptension Tech Radar! Let me know if you’d like further refinements.
