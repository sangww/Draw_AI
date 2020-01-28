int n_styles = 5;
int style = 0;
int shape_id = 0;
int n_shapes = 5;
boolean drawing = false;

boolean firstStroke = true;
boolean firstFrame = true;
float viscosity = 0.02;
float last_x = 0.0;
float last_y = 0.0;
float last_control_x = 0.0;
float last_control_y = 0.0;

float control_x = 0.0;
float control_y = 0.0;
FloatList pointx, pointy, controlx, controly;
Table table;

int id = 0;

boolean noSave = false;

float left, right, radius, middle;


void setup() {
  size(1000, 1000, P2D);
  
  radius = width * 0.3;
  middle = width / 2;
  left = middle - radius;
  right = middle + radius;
  
  table = new Table();
  table.addColumn("id");
  table.addColumn("style");
  table.addColumn("point_x");
  table.addColumn("point_y");
  table.addColumn("control_x");
  table.addColumn("control_y");
  
    drawGuideShape();
}

void drawGuideShape() {
  stroke(150,  40, 150, 100);
  //fill(150,  40, 150, 100);
  noFill();
  strokeWeight(40);
  if (shape_id == 0) {
   line(left, middle, right, middle); 
  }
  else if (shape_id == 1) {
    arc(middle, middle, radius*2, radius*2, -PI, 0);
  }
  else if (shape_id == 2) {
    line(left, middle - radius*0.5, middle, middle + radius*0.5);
    line(middle, middle + radius*0.5, right, middle - radius*0.5);
  }
  else if (shape_id == 3) {
   line(left, middle + radius*0.5, left + radius*0.25, middle - radius*0.5);
   line(left + radius*0.25, middle - radius*0.5, middle, middle + radius*0.5);
   line(middle, middle + radius*0.5, middle + radius*0.25, middle - radius*0.5);
   line(middle + radius*0.25, middle - radius*0.5, right, middle + radius*0.5);
  }
  else if (shape_id == 4) {
    arc(middle - radius*0.5, middle, radius, radius*1.5, -PI, 0);
    arc(middle + radius*0.5, middle, radius, radius*1.5, 0, PI);
  }
}

void draw() {
  if (firstFrame) {
    background(255);
    drawGuideShape();
    firstFrame = false;
  }
  
  if (drawing) {
    if (firstStroke) {
     last_x = mouseX;
     last_y = mouseY;
     firstStroke = false;
    }
    else {
     stroke(0);
     strokeWeight(3);
     line(last_x, last_y, mouseX, mouseY);
     last_x = mouseX;
     last_y = mouseY;
     
    }
    pointx.append(mouseX);
    pointy.append(mouseY);
  }
  textSize(25);
  fill(0);
  text("style " + str(style) + " / " + n_styles, 30, 30);
  if (noSave) {
   text("No Save", 30, 80); 
  }
}

void mousePressed() {
  drawing = true;
  firstStroke = true;
  pointx = new FloatList();
  pointy = new FloatList();
}

String getString(FloatList list) {
  String s = new String();
  for (int i = 0; i < list.size() - 1; i++) {
    s += str(list.get(i));
    s += " "; 
  }
  s += str(list.get(list.size()-1));
  return s;
}

void mouseReleased() {
  drawing = false;
  if (noSave) {
    noSave = false;
    return;
  }
  
  TableRow row = table.addRow();
  row.setInt("id", id);
  row.setInt("style", style);
  
  row.setString("point_x", getString(pointx));
  row.setString("point_y", getString(pointy));
  id++;
  
}

void saveData() {
    saveTable(table, str(hour()) + str(minute()) + str(second()) + ".csv");
    println("saved.");
}

void restartShape() {
  background(255);
  drawGuideShape();
}

void keyPressed() {
  int num = key - '0';
  if (num >= 0 && num < n_styles) {
    //style = num;
  }
  else if (key == 's') {
    /////
  }
  else if (key == 'd') {
   noSave = !noSave; 
  }
  else if (key == ENTER) {
    if (noSave) {
     restartShape(); 
     noSave = false;
    }
    else {
     shape_id++;
     if (shape_id >= n_shapes) {
      shape_id = 0;
      style++;
      if (style >= n_styles) {
       saveData(); 
      }
     }
     background(255);
      drawGuideShape();
    }
  }
}
