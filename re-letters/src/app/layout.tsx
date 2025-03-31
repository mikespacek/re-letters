import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import "./globals.css";

export const metadata: Metadata = {
  title: "Real Estate Data Manager - Filter Properties by Occupancy Status",
  description: "Upload CSV files containing property data and filter by owner-occupied, renter-occupied, or investor-owned status",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={GeistSans.className}>{children}</body>
    </html>
  );
}
