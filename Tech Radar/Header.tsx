import { useState } from "react";
import { Link, useLocation } from "wouter";
import vwLogo from "@assets/vw-logo.png";

export default function Header() {
  const [location] = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  return (
    <header className="bg-white dark:bg-gray-900 shadow-sm border-b border-slate-200 dark:border-gray-800 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-3 flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <img 
            src={vwLogo} 
            alt="Volkswagen Logo" 
            className="h-10 w-10"
          />
          <h1 className="text-lg md:text-xl font-bold gradient-text">VW Tech Radar</h1>
        </div>
        
        <div className="hidden md:flex items-center space-x-6">
          <nav className="flex space-x-5">
            <Link href="/" className={location === "/" ? "text-blue-600 dark:text-blue-400 font-medium" : "text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 font-medium"}>
              Radar
            </Link>
            <Link href="/technologies" className={location === "/technologies" ? "text-blue-600 dark:text-blue-400 font-medium" : "text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 font-medium"}>
              Technologies
            </Link>
          </nav>
          
          {/* <div className="relative w-64">
            <input 
              type="search" 
              placeholder="Search technologies..." 
              className="w-full pl-10 pr-4 py-2 border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-900 dark:text-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:focus:border-blue-400 dark:placeholder-gray-400" 
            />
            <svg className="w-5 h-5 text-slate-400 dark:text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
          </div> */}
        </div>
        
        <button 
          type="button" 
          className="block md:hidden text-slate-700 dark:text-slate-200" 
          aria-expanded={isMobileMenuOpen}
          onClick={toggleMobileMenu}
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16"></path>
          </svg>
        </button>
      </div>
      
      {/* Mobile menu */}
      <div className={`${isMobileMenuOpen ? 'block' : 'hidden'} md:hidden bg-white dark:bg-gray-900 border-t border-slate-200 dark:border-gray-800 px-4 py-3`}>
        <nav className="flex flex-col space-y-3">
          <Link href="/" className={location === "/" ? "text-blue-600 dark:text-blue-400 font-medium py-1" : "text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 font-medium py-1"}>
            Radar
          </Link>
          <Link href="/technologies" className={location === "/technologies" ? "text-blue-600 dark:text-blue-400 font-medium py-1" : "text-slate-600 dark:text-slate-300 hover:text-blue-600 dark:hover:text-blue-400 font-medium py-1"}>
            Technologies
          </Link>
        </nav>
        {/* <div className="mt-4 relative">
          <input 
            type="search" 
            placeholder="Search technologies..." 
            className="w-full pl-10 pr-4 py-2 border border-slate-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-slate-900 dark:text-gray-100 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:focus:border-blue-400 dark:placeholder-gray-400" 
          />
          <svg className="w-5 h-5 text-slate-400 dark:text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
          </svg>
        </div> */}
      </div>
    </header>
  );
}
