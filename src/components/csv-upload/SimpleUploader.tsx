"use client";

import React, { useState, FormEvent } from 'react';

export default function SimpleUploader() {
  const [file, setFile] = useState<File | null>(null);
  const [format, setFormat] = useState<string>('csv');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file');
      return;
    }
    
    setUploading(true);
    setError(null);
    
    try {
      console.log('Starting upload process');
      const formData = new FormData();
      formData.append('file', file);
      formData.append('output_format', format);
      
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        let message = 'Error processing file';
        try {
          const data = await response.json();
          message = data.detail || message;
        } catch (e) {
          console.error('Failed to parse error', e);
        }
        throw new Error(message);
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      
      // File extension
      const ext = format === 'csv' ? '.csv' : '.xlsx';
      const name = `${file.name.split('.')[0]}_processed${ext}`;
      
      setDownloadUrl(url);
      setFileName(name);
    } catch (err) {
      console.error('Upload error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-lg shadow-md">
      <h2 className="text-2xl font-bold mb-6">Simple File Processor</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block mb-2">Select CSV File:</label>
          <input 
            type="file" 
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="w-full p-2 border rounded"
          />
        </div>
        
        <div>
          <label className="block mb-2">Output Format:</label>
          <select 
            value={format} 
            onChange={(e) => setFormat(e.target.value)}
            className="w-full p-2 border rounded"
          >
            <option value="csv">CSV</option>
            <option value="excel">Excel</option>
            <option value="numbers">Numbers</option>
          </select>
        </div>
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}
        
        <button 
          type="submit" 
          disabled={uploading || !file} 
          className="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded disabled:opacity-50"
        >
          {uploading ? 'Processing...' : 'Process File'}
        </button>
      </form>
      
      {downloadUrl && fileName && (
        <div className="mt-6 text-center">
          <p className="text-green-600 font-medium mb-4">File processed successfully!</p>
          <a 
            href={downloadUrl} 
            download={fileName}
            className="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded inline-block"
          >
            Download File
          </a>
        </div>
      )}
    </div>
  );
} 