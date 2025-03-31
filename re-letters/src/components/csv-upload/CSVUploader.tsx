"use client";

import { useState, useRef, ChangeEvent, FormEvent } from 'react';
import { Download, FileCheck, Upload, FileType } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { FileUploadState, OutputFormat } from '@/lib/types';

const FileProcessor = () => {
  const [state, setState] = useState<FileUploadState>({
    file: null,
    isUploading: false,
    progress: 0,
    error: null,
    outputFormat: 'csv'
  });
  
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [processingComplete, setProcessingComplete] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [downloadFilename, setDownloadFilename] = useState<string | null>(null);
  
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    
    if (!selectedFile) {
      return;
    }
    
    // Validate file type
    if (!selectedFile.name.endsWith('.csv')) {
      setState(prev => ({ ...prev, error: 'Please upload a CSV file', file: null }));
      return;
    }
    
    // Validate file size (50MB max)
    if (selectedFile.size > 50 * 1024 * 1024) {
      setState(prev => ({ ...prev, error: 'File size exceeds 50MB limit', file: null }));
      return;
    }
    
    setState(prev => ({ 
      ...prev, 
      file: selectedFile,
      error: null,
      progress: 0,
      isUploading: false
    }));
    
    setProcessingComplete(false);
    setDownloadUrl(null);
  };
  
  const handleFormatChange = (value: OutputFormat) => {
    setState(prev => ({ ...prev, outputFormat: value }));
  };
  
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!state.file) {
      setState(prev => ({ ...prev, error: 'Please select a file first' }));
      return;
    }
    
    setState(prev => ({ ...prev, isUploading: true, progress: 10, error: null }));
    setProcessingComplete(false);
    
    try {
      const formData = new FormData();
      formData.append('file', state.file);
      formData.append('output_format', state.outputFormat);
      
      // Send the file to our backend API
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      
      setState(prev => ({ ...prev, progress: 75 }));
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Error processing file');
      }
      
      // Create a download URL from the response
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      
      // Determine the file extension based on the output format
      let fileExtension = '';
      switch (state.outputFormat) {
        case 'csv':
          fileExtension = '.csv';
          break;
        case 'excel':
        case 'numbers':
          fileExtension = '.xlsx';
          break;
      }
      
      // Get the original filename without extension
      const originalFilename = state.file.name.split('.')[0];
      const downloadFilename = `${originalFilename}_processed${fileExtension}`;
      
      setDownloadUrl(url);
      setDownloadFilename(downloadFilename);
      setState(prev => ({ ...prev, isUploading: false, progress: 100 }));
      setProcessingComplete(true);
      
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'An unknown error occurred',
        isUploading: false,
        progress: 0
      }));
    }
  };
  
  const handleReset = () => {
    setState({
      file: null,
      isUploading: false,
      progress: 0,
      error: null,
      outputFormat: 'csv'
    });
    
    setProcessingComplete(false);
    setDownloadUrl(null);
    
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="bg-white shadow-md rounded-lg border overflow-hidden">
      <div className="p-6 bg-gray-50 border-b">
        <div className="flex items-center gap-3">
          <FileType className="h-6 w-6 text-blue-600" />
          <div>
            <h2 className="text-xl font-semibold text-gray-800">Real Estate Data Processor</h2>
            <p className="text-sm text-gray-500">Upload, process, and export your property data</p>
          </div>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="p-6">
        <div className="space-y-6">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors bg-gray-50">
            <Input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              disabled={state.isUploading}
              className="hidden"
              id="file-upload"
            />
            <Label 
              htmlFor="file-upload" 
              className="flex flex-col items-center cursor-pointer"
            >
              <Upload className="h-12 w-12 text-blue-600 mb-3" />
              <span className="text-lg font-medium text-gray-700">Select CSV File</span>
              <span className="text-sm text-gray-500 mt-1">or drag and drop</span>
            </Label>
          </div>
          
          {state.error && (
            <Alert variant="destructive" className="bg-red-50 border-red-200 text-red-800">
              <AlertDescription>{state.error}</AlertDescription>
            </Alert>
          )}
          
          {state.file && (
            <div className="bg-blue-50 p-4 rounded-md border border-blue-100">
              <h3 className="font-medium text-blue-700 mb-2 flex items-center gap-2">
                <FileCheck className="h-4 w-4" />
                Selected file
              </h3>
              <p className="text-blue-800">{state.file.name}</p>
              <p className="text-blue-600 text-sm mt-1">Size: {(state.file.size / 1024).toFixed(2)} KB</p>
            </div>
          )}
          
          <div className="bg-gray-50 p-5 rounded-md border">
            <Label className="text-base font-medium text-gray-700 mb-3 block">Select Output Format</Label>
            <RadioGroup 
              value={state.outputFormat} 
              onValueChange={handleFormatChange}
              className="space-y-3"
            >
              <div className="flex items-center space-x-3 p-2 hover:bg-gray-100 rounded">
                <RadioGroupItem value="csv" id="csv" className="text-blue-600" />
                <Label htmlFor="csv" className="font-medium cursor-pointer">CSV (.csv)</Label>
              </div>
              <div className="flex items-center space-x-3 p-2 hover:bg-gray-100 rounded">
                <RadioGroupItem value="excel" id="excel" className="text-blue-600" />
                <Label htmlFor="excel" className="font-medium cursor-pointer">Excel (.xlsx)</Label>
              </div>
              <div className="flex items-center space-x-3 p-2 hover:bg-gray-100 rounded">
                <RadioGroupItem value="numbers" id="numbers" className="text-blue-600" />
                <Label htmlFor="numbers" className="font-medium cursor-pointer">Numbers (.xlsx)</Label>
              </div>
            </RadioGroup>
          </div>
          
          {state.isUploading && (
            <div className="space-y-2">
              <div className="text-sm flex justify-between text-gray-700">
                <span>Processing: {state.progress}%</span>
              </div>
              <Progress value={state.progress} className="h-2 w-full bg-gray-200" />
            </div>
          )}
          
          {processingComplete && downloadUrl && (
            <Alert className="bg-green-50 border-green-200 text-green-800">
              <FileCheck className="h-5 w-5 text-green-600" />
              <AlertDescription className="text-green-800 font-medium ml-2">
                File processed successfully! Click the download button below.
              </AlertDescription>
            </Alert>
          )}
        </div>
        
        <div className="mt-6 flex flex-col space-y-3">
          {!processingComplete ? (
            <Button 
              type="submit"
              disabled={!state.file || state.isUploading}
              className="w-full py-6 text-base bg-blue-600 hover:bg-blue-700"
            >
              {state.isUploading ? 'Processing...' : 'Process File'}
            </Button>
          ) : (
            <div className="flex flex-col w-full space-y-3">
              <a 
                href={downloadUrl || '#'} 
                download={downloadFilename || 'processed-file'}
                className="w-full"
              >
                <Button className="w-full py-6 text-base flex items-center justify-center gap-2 bg-green-600 hover:bg-green-700">
                  <Download className="h-5 w-5" />
                  Download {state.outputFormat.charAt(0).toUpperCase() + state.outputFormat.slice(1)} File
                </Button>
              </a>
              <Button 
                variant="outline" 
                onClick={handleReset}
                className="w-full py-5 text-base border-gray-300 text-gray-700 hover:bg-gray-50"
              >
                Process Another File
              </Button>
            </div>
          )}
        </div>
      </form>
    </div>
  );
};

export default FileProcessor; 