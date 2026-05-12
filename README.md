# Student Mental Health and Academic Performance Analysis

This project analyzes students’ their academic performance. It examines factors such as internet and electricity access, parent involvement, engagement and others, and well-being to understand how they affect learning outcomes. The goal is to gain insights that can help improve student support and academic success.

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Live Dashboard Preview</title>

<style>
  body{
    margin:0;
    background:#0f172a;
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
    font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
  }

  .frame{
    width:1100px;
    height:650px;
    position:relative;
    overflow:hidden;
    border-radius:16px;
    box-shadow:0 20px 60px rgba(0,0,0,.5);
  }

  .frame img{
    position:absolute;
    width:100%;
    height:100%;
    object-fit:cover;
    opacity:0;
    transition:opacity 1.2s ease-in-out, transform 6s ease-in-out;
    transform:scale(1.05);
  }

  .frame img.active{
    opacity:1;
    transform:scale(1);
  }

  .title{
    position:absolute;
    top:20px;
    left:30px;
    color:white;
    font-size:22px;
    letter-spacing:1px;
    opacity:.85;
  }
</style>
</head>

<body>

<div class="frame">
  <div class="title">Live Dashboard Preview</div>

  <img src="assets/dash1.png" class="active">
  <img src="assets/dash2.png">
  <img src="assets/dash3.png">
  <img src="assets/dash4.png">
</div>

<script>
  const slides = document.querySelectorAll(".frame img");
  let index = 0;

  setInterval(() => {
    slides[index].classList.remove("active");
    index = (index + 1) % slides.length;
    slides[index].classList.add("active");
  }, 3000); // 3 seconds per image
</script>

</body>
</html>
