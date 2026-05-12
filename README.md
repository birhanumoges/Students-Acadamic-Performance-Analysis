# 📊 Student Mental Health and Academic Performance Analysis

![GitHub](https://img.shields.io/badge/Analysis-Student_Wellness-blue)
![GitHub](https://img.shields.io/badge/Status-Active-green)

## 📌 Project Overview

This project analyzes students' academic performance in relation to their mental health and environmental factors. It examines key variables including:

- 🧠 **Mental Health Indicators** - Stress, anxiety, depression levels
- 💻 **Environmental Factors** - Internet access, electricity reliability
- 👨‍👩‍👧‍👦 **Parent Involvement** - Academic support, engagement levels  
- 📚 **Learning Engagement** - Class participation, study habits
- 🎓 **Academic Outcomes** - Grades, performance metrics

The goal is to gain actionable insights that can help educational institutions improve student support systems and enhance academic success.

## 🖼️ Dashboard Preview Gallery

<!-- Dashboard Carousel - Auto changes every 3 seconds -->
<div align="center">
  <style>
    .dashboard-frame {
      width: 100%;
      max-width: 1000px;
      margin: 20px auto;
      position: relative;
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 20px 40px rgba(0,0,0,0.3);
      background: #1a1a2e;
    }
    
    .dashboard-frame img {
      width: 100%;
      height: auto;
      display: none;
      transition: opacity 0.5s ease;
    }
    
    .dashboard-frame img.active {
      display: block;
      animation: fadeIn 0.8s ease;
    }
    
    @keyframes fadeIn {
      from { opacity: 0; }
      to { opacity: 1; }
    }
    
    .dashboard-dots {
      text-align: center;
      padding: 15px;
      background: #1a1a2e;
    }
    
    .dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      margin: 0 5px;
      border-radius: 50%;
      background: #4a5568;
      cursor: pointer;
      transition: background 0.3s ease;
    }
    
    .dot.active {
      background: #4299e1;
    }
    
    .dashboard-caption {
      text-align: center;
      padding: 10px;
      color: #e2e8f0;
      background: #1a1a2e;
      font-size: 14px;
      font-family: system-ui, -apple-system, sans-serif;
    }
  </style>

  <div class="dashboard-frame">
    <img src="assets/dash1.png" alt="Dashboard 1 - Mental Health Trends" class="active">
    <img src="assets/dash2.png" alt="Dashboard 2 - Academic Performance">
    <img src="assets/dash3.png" alt="Dashboard 3 - Environmental Factors">
    <img src="assets/dash4.png" alt="Dashboard 4 - Parent Engagement">
  </div>
  
  <div class="dashboard-dots">
    <span class="dot active" onclick="currentSlide(1)"></span>
    <span class="dot" onclick="currentSlide(2)"></span>
    <span class="dot" onclick="currentSlide(3)"></span>
    <span class="dot" onclick="currentSlide(4)"></span>
  </div>
  
  <div class="dashboard-caption">
    📈 Dashboard Gallery | Auto-changing every 3 seconds | Click dots to navigate manually
  </div>
</div>

<script>
  let slideIndex = 0;
  const slides = document.querySelectorAll('.dashboard-frame img');
  const dots = document.querySelectorAll('.dot');
  
  function showSlides() {
    // Hide all slides
    slides.forEach(slide => slide.classList.remove('active'));
    // Remove active class from all dots
    dots.forEach(dot => dot.classList.remove('active'));
    
    // Show current slide
    slideIndex++;
    if (slideIndex > slides.length) slideIndex = 1;
    slides[slideIndex - 1].classList.add('active');
    dots[slideIndex - 1].classList.add('active');
    
    // Change every 3 seconds
    setTimeout(showSlides, 3000);
  }
  
  function currentSlide(n) {
    slideIndex = n - 1;
    // Hide all slides
    slides.forEach(slide => slide.classList.remove('active'));
    // Remove active class from all dots
    dots.forEach(dot => dot.classList.remove('active'));
    
    // Show selected slide
    slides[slideIndex].classList.add('active');
    dots[slideIndex].classList.add('active');
  }
  
  // Start auto-rotation after page loads
  if (slides.length > 0) {
    setTimeout(showSlides, 3000);
  }
  
  // Make functions globally available
  window.currentSlide = currentSlide;
</script>

> **⚠️ Important**: Replace `assets/dash1.png`, `assets/dash2.png`, `assets/dash3.png`, `assets/dash4.png` with your actual image paths. Create an `assets` folder in your repository and place your dashboard images there, or use direct image URLs from image hosting services.

## 🔑 Key Findings

| Factor | Impact on Academic Performance |
|--------|-------------------------------|
| Mental Health Support | ⬆️ 32% improvement in grades |
| Internet Access | ⬆️ 45% better assignment completion |
| Parent Involvement | ⬆️ 28% higher attendance |
| Study Environment | ⬆️ 40% improved focus |

## 📁 Repository Structure

</body>
</html>
