CROPEYE_APP_KNOWLEDGE = """
## CropEye Mobile App — Complete Knowledge Base

### What is CropEye?
CropEye is a precision farming mobile application that helps Indian farmers monitor their fields using satellite data, AI analysis, and real-time weather. It supports English, Hindi, Marathi, and Kannada.

---

### HOME SCREEN / DASHBOARD

**Top Bar:**
- Green marquee ticker at the very top → shows LIVE MANDI PRICES from all India markets. Shows crop name, current price in ₹/quintal, price change %.
- "ACTIVE FIELD" badge (top-left) → shows current field name and live weather (temperature + condition icon)
- Profile avatar button (top-right) → opens your profile, settings, register face ID

**Map:**
- The main map shows your registered field with a coloured polygon overlay
- Google satellite imagery as base map
- Crop animation plays on your field (wheat rows, grape vineyard, mango trees etc.)
- Zooms in from earth view every time you open the app or switch fields (Google Earth animation)

**Blue navigation arrow:**
- Shows your current GPS location on the map
- Rotates to show which direction you are facing
- Tap the "548.1 KM" button to navigate to your field

**Bottom Status Bar:**
- Water droplet icon + value → recent rainfall amount
- Leaf icon + value → nitrogen status
- "Field Status" text → GOOD / IRRIGATED based on weather
- Tap it to open full Farm Insights screen

---

### FARM STATUS BUTTON (bottom-right, layers icon)
Tap to open Farm Status panel with 4 layer buttons:
1. **GROWTH** (🌱) → shows NDVI vegetation health heatmap on the map
2. **WATER** (💧) → shows water uptake / NDWI moisture heatmap
3. **SOIL** (🏔️) → shows soil analysis heatmap
4. **PESTS** (🐛) → shows pest detection risk heatmap

---

### MY FIELD BUTTON (bottom-left, tractor icon)
Opens a bottom sheet showing all your registered plots. Tap any plot to switch to it (triggers Google Earth zoom animation).

---

### CLUSTER FAB BUTTONS (right side)
**AI HELP** (✨ green button):
- Opens the AI chatbot (that's me!)

**FARM BOARD** (📊 orange button):
- Opens the Farm Board with detailed analysis panels:
  - Soil Nutrients, Pest Scanner, Crop Scanner, and more

**FARM STATUS** (layers icon):
- The 4 heatmap layer buttons

---

### SOIL NUTRIENTS PANEL
Access: Tap FARM BOARD → SOIL button
Shows:
- **NDVI button** → Vegetation Health Index (0 to 1). Values > 0.4 = healthy crops
- **NDWI button** → Water Content Index. Higher = more water in plants
- **PH** card → soil acidity (ideal: 6.0–7.5)
- **NITROGEN** card → N in kg/ha (ideal: >150)
- **PHOSPHORUS** card → P in kg/ha
- **POTASSIUM** card → K in kg/ha
- **MOISTURE** card → soil water percentage
- **CEC** card → Cation Exchange Capacity
- **OC** card → Organic Carbon percentage
- **Overall Soil Health banner** → shows GOOD/FAIR/POOR with percentage score
- Green LIVE dot on each card = data fetched from real satellite API

How to read NDVI/NDWI screens:
- Tap NDVI or NDWI button → opens full screen satellite map
How to read NDVI (Growth Map):

* Tap NDVI → opens crop growth map of your field
* Coloured overlay shows vegetation health variation across the field
* Each zone represents crop growth condition

Color meaning:
* Red / Orange → Poor growth (stressed or weak crops)
* Yellow → Moderate growth
* Light Green → Good growth
* Dark Green → Excellent growth (healthy crops)
* Score badge shows overall field growth status based on NDVI value

How to read NDWI (Water Uptake Map):
* Tap NDWI → opens water uptake map of your field
* Coloured overlay shows soil moisture / water availability variation

Color meaning:
* Dark Blue → Deficient (very low water)
* Light Blue → Less water
* Green → Adequate water
* Yellow / Orange → Excellent (optimal water level)
* Red → Excess water (over-irrigated areas)
* Score badge shows overall field water condition

- Score badge at bottom shows the field average value with label (Poor/Moderate/Good/Excellent)

---

### FARM INSIGHTS SCREEN
Access: Tap bottom status bar OR the ✨ AI HELP button area
Shows 4 analysis cards:

1. **Irrigation Depth Analysis** (Water 💧)
   - Uses water uptake satellite API
   - Shows: deficient%, less%, adequate%, excellent%, excess% pixels
   - Status: LOW / MODERATE / ADEQUATE / EXCESS
   - Tap to see detailed pixel breakdown and AI action plan

2. **Entomological Forecast** (Pest Risk 🪲)
   - Uses pest detection satellite API
   - Status: HIGH / MODERATE / LOW
   - Tap to see detailed pest risk analysis

3. **Nutrient Uptake Status** (Fertilizer 🌾)
   - Uses NPK analysis API (N, P, K values)
   - Status: RICH / READY / LOW
   - Shows actual kg/ha values for each nutrient

4. **Micro-Climate Window** (Weather ☁️)
   - Uses live OpenWeatherMap data
   - Shows current temperature, humidity, wind
   - Status: FAIR / WINDY / HOT / WET

**AI Recommendations panel** below the cards → 4 prioritised action items

**Soil Moisture Levels chart** at bottom:
- Line chart with 7 data points
- Blue bars = moisture %, purple = rainfall
- LIVE badge = real data from SAR satellite API
- Shows Low/Good/High threshold lines

---

### ADD PLOT / FIELD REGISTRATION
1. Tap MY FIELD → ADD FIELD button
2. Fill: Field name, Crop type (Wheat/Rice/Mango/Grape etc.)
3. Select crop variety
4. Set plantation date
5. Select irrigation type
6. Enter row spacing and plant spacing (optional — auto fills default for crop)
7. Draw polygon on map (tap corners of your field)
8. Submit → field saved to server
9. After saving → optional Face ID enrollment screen appears

---

### FACE ID LOGIN
- Register face during/after account creation OR from Profile Settings → "Register Face ID"
- After registering, you can log in by just scanning your face (no password needed)
- Face scan uses your front camera
- Associated with your phone number account

---

### WEATHER PANEL
Tap the weather icon/temperature in the top bar → opens weather sheet showing:
- Current temperature, feels like, humidity, wind speed, wind direction
- Cloud cover, rainfall last hour, day/night status
- Irrigation recommendation (Check field / Not needed)
- Farming tip based on current conditions
- Background gradient changes by weather condition

---

### RAIN ANIMATION
- When it's actually raining at your field location, animated rain drops appear on the map
- 3 depth layers (close/medium/far rain drops)
- Lightning flash for thunderstorms
- Rain intensity matches actual rainfall rate (mm/hour)

---

### WEATHER ALERTS BANNER
Slides in from top when dangerous conditions detected:
- ⛈️ Thunderstorm Warning
- 🌧️ Heavy Rain Alert
- 🔥 Extreme Heat (>42°C)
- 🥶 Frost Warning (<2°C)
- 💨 Severe Wind (>20 m/s)
- ☁️ Heavy Overcast
Auto-dismisses after 10 seconds, or swipe up to dismiss

---

### PROFILE & SETTINGS
Tap avatar (top-right) → Profile panel:
- Shows your name and plot count
- Language settings
- Register Face ID button
- Sign Out

---

### NAVIGATION / FOOTPRINT TRAIL
- Walk mode: as you walk your field, blue dotted trail appears on map
- Shows your GPS path as you inspect the field
- Navigation arrow rotates with your phone direction
- Distance to field shown below the arrow button

---

### CROP ANIMATIONS
Each crop type has a unique animated overlay:
- 🌾 Wheat/Barley → golden swaying stalks in rows
- 🍇 Grape → horizontal trellis rows with vine image
- 🥭 Mango → mango tree bitmap in staggered orchard grid
- 🍎 Apple → apple tree bitmap in orchard grid
- 🍌 Banana → tall banana tree bitmaps
- 🥥 Coconut → coconut palm bitmaps
- 🌱 Newly planted (<21 days) → seedling image instead of mature crop
- All crops sway with real wind speed/direction from weather API
- Custom spacing: set row spacing and plant spacing when registering field

---

### FREQUENTLY ASKED QUESTIONS

Q: How do I add a new field?
A: Tap the tractor icon (MY FIELD) at bottom-left → tap ADD FIELD → fill in details → draw polygon on map

Q: How do I see if my crops need water?
A: Look at the Farm Insights screen → Irrigation Depth Analysis card → shows % of field with low/adequate water. Or tap FARM STATUS → WATER layer to see the heatmap.

Q: What does the NDVI value mean?
A: NDVI (Normalized Difference Vegetation Index) measures plant health. 0–0.2 = bare/very stressed, 0.2–0.4 = moderate, 0.4–0.6 = healthy, 0.6–1.0 = excellent vegetation.

Q: How to switch between fields?
A: Tap MY FIELD button → tap any field name → map zooms to that field

Q: How do I check mandi prices?
A: The live ticker at the very top of the screen shows current prices. It scrolls automatically showing prices from markets across India.

Q: What is face ID login?
A: After registering, tap Profile → Register Face ID → capture 3 photos → face is linked to your account. Next time you can login without password by scanning your face.

Q: Why is my crop animation not showing?
A: Make sure you selected a crop type when registering the field. The animation appears after the field loads on the map.

Q: How accurate is the weather data?
A: Weather data comes directly from OpenWeatherMap using your field's GPS coordinates. It updates in real-time.

Q: What does the soil moisture graph show?
A: It shows daily soil moisture % for the past 7 days, with rainfall overlay. Green zone (40-80%) is optimal. Below 40% needs irrigation.
"""
