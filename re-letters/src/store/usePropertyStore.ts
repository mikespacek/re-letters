"use client";

import { create } from 'zustand';
import { PropertyData, PropertyStatus } from '@/lib/types';

interface PropertyStore {
  // Data state
  properties: PropertyData[];
  filteredProperties: PropertyData[];
  
  // Filters
  searchTerm: string;
  selectedStatus: PropertyStatus | 'All';
  sortField: keyof PropertyData;
  sortDirection: 'asc' | 'desc';
  
  // Actions
  setProperties: (data: PropertyData[]) => void;
  setSearchTerm: (term: string) => void;
  setSelectedStatus: (status: PropertyStatus | 'All') => void;
  setSortField: (field: keyof PropertyData) => void;
  toggleSortDirection: () => void;
  resetFilters: () => void;
}

const usePropertyStore = create<PropertyStore>((set, get) => ({
  // Initial state
  properties: [],
  filteredProperties: [],
  searchTerm: '',
  selectedStatus: 'All',
  sortField: 'address',
  sortDirection: 'asc',
  
  // Actions
  setProperties: (data) => {
    set({ properties: data, filteredProperties: data });
  },
  
  setSearchTerm: (term) => {
    set({ searchTerm: term });
    const { properties, selectedStatus } = get();
    set({ filteredProperties: filterAndSortProperties(properties, term, selectedStatus, get().sortField, get().sortDirection) });
  },
  
  setSelectedStatus: (status) => {
    set({ selectedStatus: status });
    const { properties, searchTerm } = get();
    set({ filteredProperties: filterAndSortProperties(properties, searchTerm, status, get().sortField, get().sortDirection) });
  },
  
  setSortField: (field) => {
    set({ sortField: field });
    const { properties, searchTerm, selectedStatus, sortDirection } = get();
    set({ filteredProperties: filterAndSortProperties(properties, searchTerm, selectedStatus, field, sortDirection) });
  },
  
  toggleSortDirection: () => {
    const newDirection = get().sortDirection === 'asc' ? 'desc' : 'asc';
    set({ sortDirection: newDirection });
    const { properties, searchTerm, selectedStatus, sortField } = get();
    set({ filteredProperties: filterAndSortProperties(properties, searchTerm, selectedStatus, sortField, newDirection) });
  },
  
  resetFilters: () => {
    set({ 
      searchTerm: '',
      selectedStatus: 'All',
      sortField: 'address',
      sortDirection: 'asc',
      filteredProperties: get().properties
    });
  }
}));

// Helper function to filter and sort properties
function filterAndSortProperties(
  properties: PropertyData[],
  searchTerm: string,
  status: PropertyStatus | 'All',
  sortField: keyof PropertyData,
  sortDirection: 'asc' | 'desc'
): PropertyData[] {
  // Filter by search term
  let filtered = properties.filter(property => {
    const matchesSearch = searchTerm === '' || 
      (property.address && property.address.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (property.mailOwnerName && property.mailOwnerName.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (property.ownerFirstName && property.ownerFirstName.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (property.ownerLastName && property.ownerLastName.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (property.standardizedHouseNumber && property.standardizedHouseNumber.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (property.streetName && property.streetName.toLowerCase().includes(searchTerm.toLowerCase()));

    const matchesStatus = status === 'All' || property.occupancyStatus === status;
      
    return matchesSearch && matchesStatus;
  });
  
  // Sort the filtered data
  return filtered.sort((a, b) => {
    const aValue = a[sortField];
    const bValue = b[sortField];
    
    // Handle different types of values
    if (typeof aValue === 'string' && typeof bValue === 'string') {
      return sortDirection === 'asc' 
        ? aValue.localeCompare(bValue)
        : bValue.localeCompare(aValue);
    }
    
    // Handle number comparison
    if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
    if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
    return 0;
  });
}

export default usePropertyStore; 