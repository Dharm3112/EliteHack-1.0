/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class', // Enable dark mode by default if class is added
    theme: {
        extend: {
            colors: {
                primary: {
                    50: '#eef2ff',
                    100: '#e0e7ff',
                    500: '#6366f1',
                    600: '#4f46e5',
                    900: '#312e81',
                },
                secondary: {
                    DEFAULT: '#1E3A8A', // The requested Dark Blue
                    hover: '#1e40af',
                    light: '#3b82f6',
                },
                dark: {
                    bg: '#0f172a',    // slate-900 background
                    surface: '#1e293b', // slate-800 surfaces
                    border: '#334155',  // slate-700 borders
                }
            },
            animation: {
                'blob': 'blob 7s infinite',
            },
            keyframes: {
                blob: {
                    '0%': { transform: 'translate(0px, 0px) scale(1)' },
                    '33%': { transform: 'translate(30px, -50px) scale(1.1)' },
                    '66%': { transform: 'translate(-20px, 20px) scale(0.9)' },
                    '100%': { transform: 'translate(0px, 0px) scale(1)' },
                }
            }
        },
    },
    plugins: [],
}
