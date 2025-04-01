import { useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import * as d3 from "d3";
import { Technology, Quadrant, Ring } from "@shared/schema";
import { 
  RING_COLORS, 
  RING_OPACITIES, 
  getRingColor, 
  getQuadrantAngle, 
  getRingBgClass,
  getQuadrantBgClass
} from "@/lib/radar-data";
import ProjectList from "./ProjectList";

// Create a simple type declaration file for this module
declare module 'd3';

export default function RadarVisualization() {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedTech, setSelectedTech] = useState<Technology | null>(null);
  const [selectedQuadrant, setSelectedQuadrant] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<Technology[] | null>(null);
  const [showProjectsList, setShowProjectsList] = useState(false);
  const [isAnimating, setIsAnimating] = useState(false); // Set animation to false by default
  const techDotsRef = useRef<{[key: number]: {element: d3.Selection<SVGCircleElement, unknown, null, undefined>, initialX: number, initialY: number, angle: number, radius: number}}>({}); 

  // Query to fetch technologies (with search param if present)
  const { 
    data: technologies = [], 
    isLoading: techsLoading,
    refetch: refetchTechnologies
  } = useQuery<Technology[]>({ 
    queryKey: ['/api/technologies', searchQuery],
    queryFn: async () => {
      const endpoint = searchQuery 
        ? `/api/technologies?q=${encodeURIComponent(searchQuery)}` 
        : '/api/technologies';
      const response = await fetch(endpoint);
      if (!response.ok) {
        throw new Error('Failed to fetch technologies');
      }
      return response.json();
    }
  });

  const { 
    data: quadrants = [], 
    isLoading: quadrantsLoading 
  } = useQuery<Quadrant[]>({ 
    queryKey: ['/api/quadrants'],
  });

  const { 
    data: rings = [], 
    isLoading: ringsLoading 
  } = useQuery<Ring[]>({ 
    queryKey: ['/api/rings'],
  });

  const isLoading = techsLoading || quadrantsLoading || ringsLoading;

  // Handle search submission
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    refetchTechnologies();
    
    if (searchQuery && technologies) {
      const results = technologies.filter((tech: Technology) => 
        tech.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tech.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (tech.tags && tech.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())))
      );
      setSearchResults(results);
    } else {
      setSearchResults(null);
    }
  };

  // Clear search results and query
  const clearSearch = () => {
    setSearchQuery('');
    setSearchResults(null);
  };

  // Select a technology from search results
  const selectTechFromSearch = (tech: Technology) => {
    setSelectedTech(tech);
    setSearchResults(null); // Hide search results after selection
    
    // If the tech is in a different quadrant than the currently selected one,
    // update the quadrant filter to show it
    if (selectedQuadrant !== null && tech.quadrant !== selectedQuadrant) {
      setSelectedQuadrant(tech.quadrant);
    }
  };

  // Toggle showing projects list
  const toggleProjectsList = () => {
    setShowProjectsList(!showProjectsList);
  };

  useEffect(() => {
    if (isLoading || !svgRef.current || !containerRef.current) {
      return;
    }

    // Clear any existing SVG content
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Get container dimensions
    const containerWidth = containerRef.current.clientWidth;
    const containerHeight = 500; // Fixed height as per design reference
    
    // Set svg dimensions
    svg
      .attr("width", containerWidth)
      .attr("height", containerHeight);

    // Create a group and center it
    const g = svg.append("g")
      .attr("transform", `translate(${containerWidth / 2}, ${containerHeight / 2})`);

    // Calculate radius based on container size
    const radius = Math.min(containerWidth, containerHeight) / 2 - 40;

    // Define ring radiuses
    const ringRadii = [
      radius * 0.25, // Adopt
      radius * 0.5,  // Trial
      radius * 0.75, // Assess
      radius         // Hold
    ];

    // Draw rings
    rings.forEach((ring: Ring, i: number) => {
      const ringColor = ring.color || RING_COLORS[i];
      
      g.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", ringRadii[i])
        .attr("fill", ringColor)
        .attr("fill-opacity", RING_OPACITIES[i])
        .attr("stroke", ringColor)
        .attr("stroke-width", 1);
      
      // Ring labels with dark mode support - using dark semi-transparent background
      const ringLabelGroup = g.append("g").attr("class", "ring-label");
      
      // Add dark background for ring label
      ringLabelGroup.append("rect")
        .attr("x", 5)
        .attr("y", -ringRadii[i] + 5)
        .attr("width", ring.name.length * 7)
        .attr("height", 20)
        .attr("rx", 4)
        .attr("fill", "rgba(13, 17, 23, 0.7)")
        .attr("class", "dark:fill-gray-900/70");
      
      // Add the ring label text
      ringLabelGroup.append("text")
        .attr("x", 10)
        .attr("y", -ringRadii[i] + 18)
        .attr("text-anchor", "start")
        .attr("fill", "#ffffff")
        .attr("class", "dark:fill-gray-200")
        .attr("font-size", "11px")
        .text(ring.name);
    });

    // Define quadrant angles
    const quadrantAngles = quadrants.map((_: Quadrant, i: number) => getQuadrantAngle(i));

    // Draw quadrant lines
    quadrantAngles.forEach((angle: number, i: number) => {
      const quadrantColor = quadrants[i]?.color || "#64748b";
      const x2 = Math.cos(angle) * radius;
      const y2 = Math.sin(angle) * radius;
      
      g.append("line")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", x2)
        .attr("y2", y2)
        .attr("stroke", quadrantColor)
        .attr("stroke-width", 1);
      
      // Quadrant labels with varying distances to match the reference image
      const labelRadiusValues = [
        radius + 35,  // Much larger distance for Languages & Frameworks (index 0)
        radius + 70,  // Adjusted for Tools (index 1)
        radius + 50,  // Adjusted for Platforms (index 2)
        radius + 170   // Adjusted for Techniques (index 3)
      ];
      const labelRadius = labelRadiusValues[i]; 
      
      // Calculate position based on fixed angles for each quadrant (45°, 135°, 225°, 315°)
      // This ensures all labels are evenly spaced at the center of each quadrant
      const quadrantCenterAngles = [
        Math.PI * 0.25,  // 45° - Languages & Frameworks
        Math.PI * 0.75,  // 135° - Tools
        Math.PI * 1.25,  // 225° - Platforms
        Math.PI * 1.82   // 315° - Techniques
      ];
      const quadrantCenterAngle = quadrantCenterAngles[i];
      
      // Position labels at the exact quadrant centers
      const labelX = Math.cos(quadrantCenterAngle) * labelRadius;
      const labelY = Math.sin(quadrantCenterAngle) * labelRadius;
      
      // Create a group for the label and its background
      const labelGroup = g.append("g").attr("class", "quadrant-label");
      
      // Add dark background for labels
      const textWidth = quadrants[i]?.name.length * 10; // Estimate text width
      labelGroup.append("rect")
        .attr("x", quadrantCenterAngle < Math.PI ? labelX - 10 : labelX - textWidth - 10)
        .attr("y", labelY - 20)
        .attr("width", textWidth + 20)
        .attr("height", 40)
        .attr("rx", 8) // Rounded corners
        .attr("fill", "rgba(13, 17, 23, 0.7)") // Dark semi-transparent background
        .attr("class", "dark:fill-gray-900/70");
      
      // Add text labels with light text
      labelGroup.append("text")
        .attr("x", labelX)
        .attr("y", labelY)
        .attr("text-anchor", quadrantCenterAngle < Math.PI ? "start" : "end")
        .attr("alignment-baseline", "middle")
        .attr("fill", "#ffffff") // White text color
        .attr("font-size", "18px")
        .attr("font-weight", "bold")
        .attr("font-family", "Arial, sans-serif")
        .attr("class", "dark:fill-gray-100")
        .text(quadrants[i]?.name || "");
    });

    // Filter technologies by selected quadrant if any
    const filteredTechs = selectedQuadrant !== null 
      ? technologies.filter((tech: Technology) => tech.quadrant === selectedQuadrant)
      : technologies;
      
    // Function to animate tech dots
    const startAnimation = () => {
      // Reset the tech dots reference
      techDotsRef.current = {};
      
      // Animation function for smooth clockwise movement
      const animateDots = () => {
        // Stop animation if component is unmounted or animation is turned off
        if (!isAnimating) return;
        
        Object.values(techDotsRef.current).forEach((dot) => {
          // Calculate new angle (clockwise rotation)
          const rotationSpeed = 0.0005; // Speed of rotation
          const newAngle = dot.angle + rotationSpeed;
          
          // Update angle for next animation frame
          dot.angle = newAngle;
          
          // Calculate new position
          const newX = Math.cos(newAngle) * dot.radius;
          const newY = Math.sin(newAngle) * dot.radius;
          
          // Update dot position
          dot.element
            .attr("cx", newX)
            .attr("cy", newY);
        });
        
        // Continue animation loop
        if (isAnimating) {
          requestAnimationFrame(animateDots);
        }
      };
      
      // Start the animation
      animateDots();
    };
    
    // Plot technologies as dots
    filteredTechs.forEach((tech: Technology) => {
      const angle = quadrantAngles[tech.quadrant] + Math.PI / 4;
      const ringRadius = ringRadii[tech.ring];
      
      // Add some randomness within the quadrant and ring
      const randomAngleOffset = (Math.random() - 0.5) * Math.PI / 3;
      const randomRadiusScale = 0.7 + Math.random() * 0.3;
      
      const finalAngle = angle + randomAngleOffset;
      const finalRadius = ringRadius * randomRadiusScale;
      
      const x = Math.cos(finalAngle) * finalRadius;
      const y = Math.sin(finalAngle) * finalRadius;
      
      // Check if this technology is the selected one
      const isSelected = selectedTech && selectedTech.id === tech.id;
      
      // Technology dot with dark mode support - increased size for better visibility without labels
      const techDot = g.append("circle")
        .attr("cx", x)
        .attr("cy", y)
        .attr("r", isSelected ? 9 : 7) // Larger dots since we don't have labels
        .attr("fill", rings[tech.ring]?.color || RING_COLORS[tech.ring])
        .attr("stroke", isSelected ? "#000" : "#fff")
        .attr("stroke-width", isSelected ? 3 : 2) // Thicker stroke for better visibility
        .attr("class", "tech-dot dark:stroke-gray-800")
        .attr("cursor", "pointer")
        .attr("data-tech-id", tech.id) // Add data attribute for identification
        .attr("data-tech-name", tech.name) // Add data attribute for tooltip/accessibility
        .on("mouseover", function(this: SVGCircleElement) {
          // Pause animation when hovering
          d3.selectAll(".tech-dot").interrupt();
          
          const isDarkMode = document.documentElement.classList.contains('dark');
          d3.select(this)
            .attr("r", 10) // Even larger on hover
            .attr("stroke", isDarkMode ? "#fff" : "#000")
            .attr("stroke-width", 3);
            
          // Add a temporary tooltip on hover
          const tooltip = g.append("text")
            .attr("x", x)
            .attr("y", y - 15)
            .attr("text-anchor", "middle")
            .attr("fill", isDarkMode ? "#fff" : "#000")
            .attr("class", `tech-tooltip dark:fill-gray-100`)
            .attr("font-size", "12px")
            .attr("font-weight", "600")
            .attr("pointer-events", "none") // Don't interfere with mouse events
            .attr("filter", "drop-shadow(0px 0px 3px rgba(255,255,255,0.8)) drop-shadow(0px 0px 1px rgba(0,0,0,0.5))")
            .text(tech.name);
            
          // Add light background behind tooltip text
          const bbox = tooltip.node()?.getBBox();
          if (bbox) {
            g.insert("rect", "text.tech-tooltip")
              .attr("x", bbox.x - 5)
              .attr("y", bbox.y - 2)
              .attr("width", bbox.width + 10)
              .attr("height", bbox.height + 4)
              .attr("rx", 3)
              .attr("fill", isDarkMode ? "rgba(30, 41, 59, 0.8)" : "rgba(255, 255, 255, 0.8)")
              .attr("class", "tooltip-bg")
              .attr("pointer-events", "none");
          }
        })
        .on("mouseout", function(this: SVGCircleElement) {
          if (!isSelected) {
            const isDarkMode = document.documentElement.classList.contains('dark');
            d3.select(this)
              .attr("r", 7)
              .attr("stroke", isDarkMode ? "#333" : "#fff")
              .attr("stroke-width", 2);
          }
          // Remove all temporary tooltips
          d3.selectAll(".tech-tooltip, .tooltip-bg").remove();
          
          // Resume animation if it was paused
          if (isAnimating) {
            startAnimation();
          }
        })
        .on("click", function() {
          setSelectedTech(tech);
        });
        
      // Store the dot reference with its initial position and angle for animation
      techDotsRef.current[tech.id] = {
        element: techDot,
        initialX: x,
        initialY: y,
        angle: finalAngle,
        radius: finalRadius
      };
    });
    
    // Start animation for all dots
    if (isAnimating) {
      startAnimation();
    }

  }, [isLoading, technologies, quadrants, rings, selectedQuadrant, selectedTech, isAnimating]);
  
  // Effect to handle animation state changes
  useEffect(() => {
    if (isAnimating && Object.keys(techDotsRef.current).length > 0) {
      // Start animation when state changes to true and we have dots
      const animateDots = () => {
        if (!isAnimating) return;
        
        Object.values(techDotsRef.current).forEach((dot) => {
          // Calculate new angle (clockwise rotation)
          const rotationSpeed = 0.0005; // Speed of rotation
          const newAngle = dot.angle + rotationSpeed;
          
          // Update angle for next animation frame
          dot.angle = newAngle;
          
          // Calculate new position
          const newX = Math.cos(newAngle) * dot.radius;
          const newY = Math.sin(newAngle) * dot.radius;
          
          // Update dot position
          dot.element
            .attr("cx", newX)
            .attr("cy", newY);
        });
        
        // Continue animation loop
        if (isAnimating) {
          requestAnimationFrame(animateDots);
        }
      };
      
      // Start the animation
      requestAnimationFrame(animateDots);
    }
  }, [isAnimating]);

  const handleQuadrantFilter = (quadrantId: number | null) => {
    setSelectedQuadrant(quadrantId);
  };

  return (
    <div className="bg-background radar-bg rounded-lg shadow-sm border border-gray-200 dark:bg-gray-900 dark:border-gray-700 p-6 mb-8 container mx-auto">
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center mb-6">
        {/* <h2 className="text-xl font-bold mb-2 lg:mb-0 gradient-text dark:bg-gradient-to-r dark:from-blue-300 dark:to-indigo-200">Technology Radar Visualization</h2> */}
        
        <div className="w-full lg:w-auto flex flex-col sm:flex-row gap-4">
          {/* Search component with dark mode support */}
          <form onSubmit={handleSearch} className="relative w-full lg:w-64">
            <input 
              type="search" 
              placeholder="Search technologies..." 
              className="w-full pl-10 pr-10 py-2 border border-slate-300 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100 dark:placeholder-gray-400 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-blue-500 dark:focus:border-blue-400" 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            <svg 
              className="w-5 h-5 text-slate-400 dark:text-gray-400 absolute left-3 top-2.5" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
            {searchQuery && (
              <button
                type="button"
                onClick={clearSearch}
                className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-600 dark:text-gray-400 dark:hover:text-gray-200"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
            
            {/* Search results dropdown */}
            {searchResults && searchResults.length > 0 && (
              <div className="absolute z-10 mt-1 w-full bg-background dark:bg-gray-800 rounded-md shadow-lg border border-slate-200 dark:border-gray-700 max-h-60 overflow-auto">
                <ul className="py-1">
                  {searchResults.map((tech: Technology) => (
                    <li 
                      key={tech.id} 
                      className="px-4 py-2 hover:bg-slate-100 dark:hover:bg-gray-700 cursor-pointer"
                      onClick={() => selectTechFromSearch(tech)}
                    >
                      <div className="font-medium">{tech.name}</div>
                      <div className="flex items-center mt-1 space-x-2">
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getRingBgClass(tech.ring)}`}>
                          {rings && rings[tech.ring]?.name}
                        </span>
                        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getQuadrantBgClass(tech.quadrant)}`}>
                          {quadrants && quadrants[tech.quadrant]?.name}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            {searchResults && searchResults.length === 0 && (
              <div className="absolute z-10 mt-1 w-full bg-background dark:bg-gray-800 rounded-md shadow-lg border border-slate-200 dark:border-gray-700">
                <div className="px-4 py-3 text-sm text-slate-500 dark:text-slate-400">No technologies found</div>
              </div>
            )}
          </form>
          
          {/* Quadrant filter buttons with dark mode support */}
          <div className="flex flex-wrap gap-2">
            <button 
              className={`${selectedQuadrant === null 
                ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' 
                : 'bg-slate-100 hover:bg-blue-100 text-slate-800 hover:text-blue-800 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-blue-900 dark:hover:text-blue-200'
              } px-3 py-1.5 rounded-full text-sm font-medium`}
              onClick={() => handleQuadrantFilter(null)}
            >
              All Quadrants
            </button>
            {quadrants.map((quadrant: Quadrant, index: number) => (
              <button 
                key={quadrant.id}
                className={`${selectedQuadrant === index 
                  ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' 
                  : 'bg-slate-100 hover:bg-blue-100 text-slate-800 hover:text-blue-800 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-blue-900 dark:hover:text-blue-200'
                } px-3 py-1.5 rounded-full text-sm font-medium`}
                onClick={() => handleQuadrantFilter(index)}
              >
                {quadrant.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-8">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      ) : (
        <div className="flex flex-col">
          <div className="flex flex-col lg:flex-row mb-6">
            <div className="w-full lg:w-2/3 mb-6 lg:mb-0">
              <div ref={containerRef} className="w-full h-[500px] flex items-center justify-center">
                <svg ref={svgRef}></svg>
              </div>
            </div>
            <div className="w-full lg:w-1/3 lg:pl-6">
              <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2 dark:text-gray-100">Selected Technology</h3>
                <div className="bg-slate-50 dark:bg-gray-800 rounded-lg p-4 border border-slate-200 dark:border-gray-700 tech-card shadow-sm dark:shadow-gray-900/30">
                  {selectedTech ? (
                    <div>
                      <div className="mb-2">
                        <h4 className="font-semibold text-lg text-gray-900 dark:text-gray-100">{selectedTech.name}</h4>
                        <div className="flex items-center flex-wrap gap-2 mt-2">
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getRingBgClass(selectedTech.ring)}`}>
                            {rings[selectedTech.ring]?.name}
                          </span>
                          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${getQuadrantBgClass(selectedTech.quadrant)}`}>
                            {quadrants[selectedTech.quadrant]?.name}
                          </span>
                          {selectedTech.tags && selectedTech.tags.map((tag, index) => (
                            <span key={index} className="px-2 py-0.5 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-300">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                      <p className="text-slate-600 dark:text-slate-300 text-sm mb-3 leading-relaxed">{selectedTech.description}</p>
                      
                      {selectedTech.website && (
                        <div className="mb-2">
                          <a 
                            href={selectedTech.website} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="text-blue-600 dark:text-blue-400 text-sm hover:underline flex items-center"
                          >
                            <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                            </svg>
                            Visit website
                          </a>
                        </div>
                      )}
                      
                      <div className="mt-3 pt-3 border-t border-slate-200 dark:border-gray-700 flex justify-between">
                        <a href={`/technologies?tech=${selectedTech.id}`} className="text-blue-600 dark:text-blue-400 text-sm font-medium hover:text-blue-800 dark:hover:text-blue-300">
                          View full details →
                        </a>
                        <button 
                          onClick={toggleProjectsList}
                          className="text-blue-600 dark:text-blue-400 text-sm font-medium hover:text-blue-800 dark:hover:text-blue-300 flex items-center"
                        >
                          Related Projects
                          <svg className={`w-4 h-4 ml-1 transform ${showProjectsList ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ) : (
                    <p className="text-slate-500 dark:text-slate-400 text-sm">Click on a technology in the radar to see details</p>
                  )}
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold mb-2 dark:text-gray-100">Legend</h3>
                <div className="space-y-2 p-2 rounded-md bg-white/50 dark:bg-gray-800/80 border border-gray-100 dark:border-gray-700">
                  {rings.map((ring: Ring, index: number) => (
                    <div key={ring.id} className="flex items-center">
                      <div 
                        className="w-4 h-4 rounded-full mr-2 border border-gray-200 dark:border-gray-600" 
                        style={{ backgroundColor: ring.color || RING_COLORS[index] }}
                      ></div>
                      <span className="text-sm dark:text-gray-200">{ring.name}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          
          {/* Projects List */}
          {showProjectsList && (
            <ProjectList selectedTechnology={selectedTech} />
          )}
        </div>
      )}
    </div>
  );
}
