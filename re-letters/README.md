# Real Estate Data Manager

A modern web application that allows users to upload CSV files containing real estate property data and filter by occupancy status (Owner-Occupied, Renter-Occupied, Investor-Owned).

## Features

- **CSV File Upload**: Upload and parse CSV files containing property data
- **Automatic Status Detection**: Intelligently determines property occupancy status based on available data
- **Filtering System**: Filter properties by:
  - Owner-Occupied (primary residence)
  - Renter-Occupied (leased to tenants)
  - Investor-Owned (vacant, secondary home, or rental)
- **Search & Sort**: Search by property address or owner name, sort by any column
- **Export Functionality**: Download filtered data as a CSV file
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

- **Frontend Framework**: Next.js 13+ (App Router)
- **UI Components**: Shadcn UI with Tailwind CSS
- **CSV Parsing**: PapaParse
- **State Management**: Zustand
- **TypeScript**: For type safety and better developer experience

## Getting Started

### Prerequisites

- Node.js 18.0.0 or later
- npm or yarn package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/real-estate-data-manager.git
   cd real-estate-data-manager
   ```

2. Install dependencies:
   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:
   ```bash
   npm run dev
   # or
   yarn dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

## CSV File Format

The application works best with CSV files that include the following columns (though it can adapt to various formats):

- `address` - Property street address
- `city` - City name
- `state` - State abbreviation
- `zip` - ZIP/Postal code
- `owner_name` - Property owner's name
- `occupancy_status` - Current occupancy status (if available)
- `mailing_address` - Owner's mailing address (used to determine if investor-owned)
- `purchase_price` - Property purchase price
- `bedrooms` - Number of bedrooms
- `bathrooms` - Number of bathrooms
- `square_feet` - Property square footage
- `year_built` - Year the property was built

A sample CSV file is included in the `/public` directory.

## How Occupancy Detection Works

The application determines occupancy status using the following logic:

1. If an `occupancy_status` column exists, it will use that value directly
2. If the property address and mailing address don't match, it's likely an investment property
3. Keywords in status fields like "owner", "renter", "tenant", "invest", "vacant" are used to determine status

## Deployment

This is a Next.js application that can be deployed to various platforms:

### Vercel (Recommended)
```bash
npm install -g vercel
vercel
```

### Netlify
```bash
npm install -g netlify-cli
netlify deploy
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Next.js](https://nextjs.org/) - The React Framework
- [Shadcn UI](https://ui.shadcn.com/) - Beautifully designed components
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [PapaParse](https://www.papaparse.com/) - Fast CSV parser for JavaScript
- [Zustand](https://github.com/pmndrs/zustand) - State management solution
