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
                framer: {
                    canvas: '#000000',
                    panel: '#0F0F0F',
                    active: '#1C1C1C',
                    border: '#1A1A1A',
                    blue: '#0099FF',
                    text: '#FFFFFF',
                    muted: '#8E8E8E',
                    dim: '#555555'
                }
            },
            borderRadius: {
                'panel': '16px',
                'pill': '999px',
                'menu': '8px'
            },
            fontFamily: {
                sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
