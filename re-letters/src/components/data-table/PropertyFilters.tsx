"use client";

import { useState, useEffect } from 'react';
import { Search, X } from 'lucide-react';

import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { PropertyStatus } from '@/lib/types';
import usePropertyStore from '@/store/usePropertyStore';

const PropertyFilters = () => {
  const { 
    searchTerm, 
    setSearchTerm,
    selectedStatus, 
    setSelectedStatus,
    resetFilters,
    properties
  } = usePropertyStore();
  
  // Local state for input field
  const [tempSearchTerm, setTempSearchTerm] = useState(searchTerm);
  
  // Update local state when store changes
  useEffect(() => {
    setTempSearchTerm(searchTerm);
  }, [searchTerm]);
  
  // Counts for each status type
  const statusCounts = {
    'All': properties.length,
    'Owner-Occupied': properties.filter(p => p.occupancyStatus === 'Owner-Occupied').length,
    'Renter-Occupied': properties.filter(p => p.occupancyStatus === 'Renter-Occupied').length,
    'Investor-Owned': properties.filter(p => p.occupancyStatus === 'Investor-Owned').length,
    'Unknown': properties.filter(p => p.occupancyStatus === 'Unknown').length
  };
  
  // Handle search form submission
  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchTerm(tempSearchTerm);
  };
  
  // Handle status change
  const handleStatusChange = (value: string) => {
    setSelectedStatus(value as PropertyStatus | 'All');
  };
  
  // Clear search
  const clearSearch = () => {
    setTempSearchTerm('');
    setSearchTerm('');
  };
  
  return (
    <Card className="w-full">
      <CardContent className="pt-6">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {/* Search Input */}
          <form onSubmit={handleSearch} className="flex w-full items-center space-x-2">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search address or owner..."
                value={tempSearchTerm}
                onChange={(e) => setTempSearchTerm(e.target.value)}
                className="w-full pl-9"
              />
              {tempSearchTerm && (
                <button
                  type="button"
                  onClick={clearSearch}
                  className="absolute right-2.5 top-2.5 text-muted-foreground hover:text-foreground"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
            <Button type="submit" variant="default" size="sm">
              Search
            </Button>
          </form>
          
          {/* Status Filter */}
          <div className="flex flex-col space-y-1">
            <label className="text-sm font-medium text-muted-foreground">Occupancy Status</label>
            <Select
              value={selectedStatus}
              onValueChange={handleStatusChange}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="All">All Properties ({statusCounts['All']})</SelectItem>
                <SelectItem value="Owner-Occupied">Owner-Occupied ({statusCounts['Owner-Occupied']})</SelectItem>
                <SelectItem value="Renter-Occupied">Renter-Occupied ({statusCounts['Renter-Occupied']})</SelectItem>
                <SelectItem value="Investor-Owned">Investor-Owned ({statusCounts['Investor-Owned']})</SelectItem>
                <SelectItem value="Unknown">Unknown Status ({statusCounts['Unknown']})</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* Reset Button */}
          <div className="flex items-end">
            <Button 
              variant="outline" 
              onClick={resetFilters}
              className="w-full md:w-auto"
            >
              Reset Filters
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default PropertyFilters; 