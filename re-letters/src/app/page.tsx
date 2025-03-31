import FileProcessor from '@/components/csv-upload/CSVUploader';
import Link from 'next/link';
import { ExternalLink, Download, FileType, BarChart3 } from 'lucide-react';

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50">
      <header className="bg-white border-b shadow-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <FileType className="h-7 w-7 text-blue-600" />
            <h1 className="text-xl font-bold">RE Data Processor</h1>
          </div>
          <nav>
            <Link 
              href="/sample-data.csv" 
              className="text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
              download
            >
              <Download className="h-4 w-4" />
              Download Sample CSV
            </Link>
          </nav>
        </div>
      </header>
      
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8 max-w-3xl mx-auto">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-800">Real Estate Data Processor</h2>
            <p className="text-gray-600 mt-3 text-lg">
              Upload your CSV files, standardize data formatting, and export to your preferred format
            </p>
          </div>
        </div>
        
        <div className="max-w-4xl mx-auto">
          <FileProcessor />
          
          <div className="mt-8 bg-white p-6 rounded-lg shadow border">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-blue-600" />
              Processing Features
            </h3>
            
            <ul className="space-y-3 text-gray-700">
              <li className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-blue-600"></div>
                <span>Standardizes ZIP codes to 5 digits with leading zeros</span>
              </li>
              <li className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-blue-600"></div>
                <span>Formats phone numbers consistently</span>
              </li>
              <li className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-blue-600"></div>
                <span>Cleans price fields (removes $ and commas)</span>
              </li>
              <li className="flex items-center gap-2">
                <div className="h-2 w-2 rounded-full bg-blue-600"></div>
                <span>Strips whitespace from all fields</span>
              </li>
            </ul>
          </div>
        </div>
      </div>
      
      <footer className="mt-12 border-t py-8 bg-white">
        <div className="container mx-auto px-4 text-center text-gray-600">
          <p>Real Estate Data Processor &copy; {new Date().getFullYear()}</p>
        </div>
      </footer>
    </main>
  );
}
