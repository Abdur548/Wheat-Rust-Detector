/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#1D6A3E',
        'primary-light': '#2E9E5A',
        accent: '#F0A500',
        background: '#F8FAF9',
        card: '#FFFFFF',
        'text-primary': '#1A2E1A',
        'text-secondary': '#6B7280',
        success: '#10B981',
        danger: '#EF4444',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
