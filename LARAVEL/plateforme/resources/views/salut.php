<!DOCTYPE HTML>
<html>
<body>
<h1 >SALUT ! <?php echo $nom ." ".$prenom; ?></h1>
<h2> Cours : </h2>
<ul>
  <?php foreach($cours as $value): ?>
    <li><?php echo $value ?></li>
  <?php endforeach; ?>
</ul>
</body>
</html>