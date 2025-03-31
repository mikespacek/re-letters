import React, { FormEvent } from 'react';

const CSVUploader: React.FC = () => {
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!state.file) {
      setState(prev => ({ ...prev, error: 'Please select a file first' }));
      return;
    }
    
    setState(prev => ({ ...prev, isUploading: true, progress: 10, error: null }));
    setProcessingComplete(false);
    
    try {
      console.log('Starting file upload process', state.file.name, state.outputFormat);
      const formData = new FormData();
      formData.append('file', state.file);
      formData.append('output_format', state.outputFormat);
      
      console.log('Sending request to backend API');
      // Send the file to our backend API with a timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
      
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      console.log('Response received', response.status);
      setState(prev => ({ ...prev, progress: 75 }));
      
      if (!response.ok) {
        let errorMessage = 'Error processing file';
        try {
          const errorData = await response.json();
          errorMessage = errorData.detail || errorMessage;
        } catch (e) {
          console.error('Failed to parse error response', e);
        }
        throw new Error(errorMessage);
      }
      
      console.log('Processing response blob');
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
      console.log('File processing complete');
      
    } catch (error) {
      console.error('Error during file processing:', error);
      setState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'An unknown error occurred',
        isUploading: false,
        progress: 0
      }));
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* Form fields and submit button */}
    </form>
  );
};

export default CSVUploader; 