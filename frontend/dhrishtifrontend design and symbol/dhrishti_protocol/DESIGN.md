---
name: Dhrishti Protocol
colors:
  surface: '#121212'
  surface-dim: '#131313'
  surface-bright: '#3a3939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353534'
  on-surface: '#e5e2e1'
  on-surface-variant: '#c8c8ac'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#929279'
  outline-variant: '#474833'
  surface-tint: '#c3d000'
  primary: '#ffffff'
  on-primary: '#2f3300'
  primary-container: '#dfed1a'
  on-primary-container: '#636900'
  inverse-primary: '#5d6300'
  secondary: '#a7caf3'
  on-secondary: '#063254'
  secondary-container: '#274b6e'
  on-secondary-container: '#99bbe4'
  tertiary: '#ffffff'
  on-tertiary: '#00363b'
  tertiary-container: '#98f1fa'
  on-tertiary-container: '#006f77'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#dfed1a'
  primary-fixed-dim: '#c3d000'
  on-primary-fixed: '#1b1d00'
  on-primary-fixed-variant: '#454a00'
  secondary-fixed: '#d0e4ff'
  secondary-fixed-dim: '#a7caf3'
  on-secondary-fixed: '#001d35'
  on-secondary-fixed-variant: '#25496c'
  tertiary-fixed: '#98f1fa'
  tertiary-fixed-dim: '#7bd4dd'
  on-tertiary-fixed: '#002022'
  on-tertiary-fixed-variant: '#004f55'
  background: '#131313'
  on-background: '#e5e2e1'
  surface-variant: '#353534'
  border: '#27272A'
  text-primary: '#EDEDED'
  text-secondary: '#A1A1AA'
  success: '#10B981'
  critical: '#EF4444'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: '1.2'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.5'
  label-mono:
    fontFamily: Geist Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.0'
    letterSpacing: 0.05em
  data-technical:
    fontFamily: Geist Mono
    fontSize: 13px
    fontWeight: '400'
    lineHeight: '1.4'
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.2'
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin-page: 24px
  panel-gap: 1px
  container-max: 1920px
---

## Brand & Style

The design system is engineered for high-stakes surveillance and technical intelligence. It adopts a **Stark Technical** aesthetic, drawing inspiration from the precision of Linear and the analytical depth of Palantir. The interface prioritizes signal over noise, utilizing a strict dark-mode environment that reduces eye strain during long-duration monitoring.

The style is characterized by a "Linear" aesthetic: razor-sharp borders, monochromatic surfaces, and high-contrast data visualization. Interactivity is conveyed through "glow" states—subtle border illuminations and spotlight effects—simulating a high-end hardware console. The emotional response is one of absolute control, precision, and mission-critical reliability.

## Colors

The palette is anchored in a true-black `#0A0A0A` foundation to ensure maximum panel contrast. The primary brand color is a high-visibility sulfur yellow (`#E4F222`), used sparingly for critical alerts, active states, and tactical highlights. A technical blue (`#B2D5FF`) serves as the secondary accent for data overlays and non-critical interactive elements.

Surfaces use `#121212` to create depth without relying on shadows. Borders are strictly defined at `#27272A`, creating a grid-like structure that reinforces the technical nature of the application. Status indicators follow industry-standard semantic patterns: critical failures in red and operational status in green, but adjusted for high-contrast visibility against the dark backdrop.

## Typography

Typography is bifurcated by function. **Inter** handles all UI controls, navigation, and structural labels, providing a modern and highly legible interface. **Geist Mono** is reserved for all technical data, including timestamps, coordinate systems, IP addresses, and log entries.

All monospaced text is set with slightly increased letter spacing to enhance readability in high-density data views. Headlines use tighter tracking and heavier weights to establish a clear hierarchy against the secondary text. Use `label-mono` for all metadata descriptions and `data-technical` for the actual streaming values.

## Layout & Spacing

This design system utilizes a **Fixed Grid** model based on a 4px atomic unit. The layout is optimized for 16:9 displays, common in surveillance command centers. Panels are separated by 1px borders rather than wide gutters, maximizing screen real estate for video feeds and telemetry.

The layout adapts to three primary states:
- **Command View (Desktop):** A multi-panel dashboard with sidebar navigation and a persistent bottom status bar.
- **Analysis View (Tablet):** A focused dual-pane layout for deep-diving into specific data threads.
- **Field View (Mobile):** A stacked single-column feed with collapsible controls.

Margins are kept tight (24px) to emphasize the "edge-to-edge" technical feel.

## Elevation & Depth

Depth is conveyed through **Tonal Layering** and **Luminous Borders** rather than traditional shadows.
- **Level 0 (Background):** `#0A0A0A` – The base canvas.
- **Level 1 (Panels):** `#121212` – Used for the primary workspace and sidebars.
- **Level 2 (Popovers/Modals):** `#1C1C1C` – Elevated surfaces with a 1px border of `#3F3F46`.

**Special Effects:**
- **Border Glow:** Interactive elements (like active video feeds) use a 1px inner glow of the primary color.
- **Glare Effect:** Hovering over cards or buttons triggers a subtle diagonal light sweep (gradient overlay) to simulate a glass-screen interaction.
- **Spotlight:** On hover, a faint radial gradient follows the cursor behind transparent layers to highlight proximity.

## Shapes

The shape language is predominantly sharp to reinforce the military-grade, technical aesthetic. Standard UI components use a `0.25rem` (Soft) radius. Large containers or video feed cards may scale up to `0.5rem` (rounded-lg) as a maximum. Buttons and input fields should remain strictly at the base roundedness to maintain a modular, "blocked-in" appearance.

## Components

### Buttons
- **Primary:** Background `#E4F222`, Text `#0A0A0A`. No border. High-gloss hover effect.
- **Tactical:** Background transparent, 1px border `#27272A`, Text `#EDEDED`. On hover, border turns `#B2D5FF` with a subtle outer glow.

### Input Fields
- Dark backgrounds (`#0A0A0A`) with a 1px border. Focus state changes border to the primary color and adds a "scanline" animation inside the field.

### Status Chips
- Small, rectangular blocks using `label-mono`. Use high-saturation semantic colors for the text and a low-opacity version of the same color for the background (e.g., Critical: Red text on 10% Red background).

### Video Feed Cards
- Sharp corners, 1px border. Top-right overlay contains Geist Mono timestamps. On hover, the border illuminates to signify selection.

### Technical Lists
- High-density rows with 1px bottom dividers. Alternating "zebra" stripes using `#161616` and `#121212` for multi-column data clarity.

### Control HUD
- A floating toolbar at the bottom of the screen using a backdrop blur (glassmorphism) of 20px and a semi-transparent `#121212` background.