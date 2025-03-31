"use client";

import { useState } from 'react';
import { ArrowUpDown, Download } from 'lucide-react';
import Papa from 'papaparse';

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { PropertyData } from '@/lib/types';
import usePropertyStore from '@/store/usePropertyStore';

// Display column configuration
interface ColumnConfig {
  key: keyof PropertyData;
  label: string;
  sortable: boolean;
  render?: (value: any, property: PropertyData) => React.ReactNode;
}

const PropertyTable = () => {
  const { 
    filteredProperties, 
    sortField, 
    sortDirection, 
    setSortField, 
    toggleSortDirection 
  } = usePropertyStore();
  
  const [currentPage, setCurrentPage] = useState(1);
  const rowsPerPage = 10;
  
  // Define the columns to display
  const columns: ColumnConfig[] = [
    {
      key: 'address',
      label: 'Property Address',
      sortable: true,
      render: (value, property) => (
        <div>
          <div className="font-medium">{value}</div>
          <div className="text-sm text-muted-foreground">
            {property.zipCode ? `${property.state || ''} ${property.zipCode}`.trim() : ''}
          </div>
        </div>
      ),
    },
    {
      key: 'mailOwnerName',
      label: 'Owner',
      sortable: true,
      render: (value, property) => (
        <div>
          {value || (
            <span>
              {property.ownerFirstName} {property.ownerLastName}
            </span>
          )}
        </div>
      ),
    },
    {
      key: 'occupancyStatus',
      label: 'Occupancy',
      sortable: true,
      render: (value) => {
        let statusColor = 'bg-gray-100 text-gray-800';
        if (value === 'Owner-Occupied') statusColor = 'bg-green-100 text-green-800';
        if (value === 'Renter-Occupied') statusColor = 'bg-blue-100 text-blue-800';
        if (value === 'Investor-Owned') statusColor = 'bg-amber-100 text-amber-800';
        
        return (
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColor}`}>
            {value}
          </span>
        );
      }
    },
    {
      key: 'taxBillingAddress',
      label: 'Tax Billing Address',
      sortable: true,
    },
    {
      key: 'subdivision',
      label: 'Subdivision',
      sortable: true,
    },
    {
      key: 'mlsSaleDate',
      label: 'Sale Date',
      sortable: true,
    },
    {
      key: 'mlsSaleClosePrice',
      label: 'Sale Price',
      sortable: true,
      render: (value) => value ? `$${Number(value).toLocaleString()}` : '-',
    },
    {
      key: 'exemptions',
      label: 'Exemptions',
      sortable: true,
    },
  ];
  
  // Sort handler
  const handleSort = (field: keyof PropertyData) => {
    if (sortField === field) {
      toggleSortDirection();
    } else {
      setSortField(field);
    }
  };
  
  // Calculate pagination
  const totalPages = Math.ceil(filteredProperties.length / rowsPerPage);
  const startIndex = (currentPage - 1) * rowsPerPage;
  const paginatedData = filteredProperties.slice(startIndex, startIndex + rowsPerPage);
  
  // CSV Export function
  const handleExportCSV = () => {
    if (filteredProperties.length === 0) return;
    
    const csv = Papa.unparse(filteredProperties);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'property_data_export.csv');
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>Property List</CardTitle>
        <Button 
          variant="outline" 
          size="sm" 
          onClick={handleExportCSV}
          disabled={filteredProperties.length === 0}
        >
          <Download className="h-4 w-4 mr-2" />
          Export CSV
        </Button>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                {columns.map((column) => (
                  <TableHead key={column.key.toString()}>
                    {column.sortable ? (
                      <button
                        className="inline-flex items-center space-x-1"
                        onClick={() => handleSort(column.key)}
                      >
                        <span>{column.label}</span>
                        <ArrowUpDown className={`h-4 w-4 ${sortField === column.key ? 'text-foreground' : 'text-muted-foreground'}`} />
                      </button>
                    ) : (
                      column.label
                    )}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedData.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-24 text-center">
                    No properties found.
                  </TableCell>
                </TableRow>
              ) : (
                paginatedData.map((property) => (
                  <TableRow key={property.id}>
                    {columns.map((column) => (
                      <TableCell key={`${property.id}-${column.key.toString()}`}>
                        {column.render 
                          ? column.render(property[column.key], property)
                          : property[column.key] !== undefined
                            ? String(property[column.key])
                            : '-'
                        }
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
      {totalPages > 1 && (
        <CardFooter className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Showing {startIndex + 1}-{Math.min(startIndex + rowsPerPage, filteredProperties.length)} of {filteredProperties.length}
          </div>
          <div className="flex space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage((prev) => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
            >
              Previous
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage((prev) => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
            >
              Next
            </Button>
          </div>
        </CardFooter>
      )}
    </Card>
  );
};

export default PropertyTable; 