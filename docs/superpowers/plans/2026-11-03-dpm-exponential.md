# DPM++ 2M and Euler a Exponential

1. Capture NUI/Diffusers schedule, timestep and warning fixtures for Exponential at several step counts with a leading-spacing checkpoint. Add failing comparisons and worker tests.
2. Add DPM++ 2M exponential sigma layout with Diffusers integer timestep conversion, and route all generation passes through it. Make Euler a follow NUI's warning and checkpoint-spacing fallback.
3. Run the independent fixture and worker tests, including Hires and regression schedules, and rebuild the native worker distribution.
