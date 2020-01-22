int n_styles = 3;
int style = 0;
boolean drawing = false;

boolean firstStroke = true;
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


void setup() {
  size(1000, 1000, P2D);
  table = new Table();
  table.addColumn("id");
  table.addColumn("style");
  table.addColumn("point_x");
  table.addColumn("point_y");
  table.addColumn("control_x");
  table.addColumn("control_y");
  
}

void draw() {
  
  if (drawing) {
    if (firstStroke) {
     last_x = mouseX;
     last_y = mouseY;
     control_x = mouseX;
     control_y = mouseY;
     last_control_x = mouseX;
     last_control_y = mouseY;
     firstStroke = false;
    }
    else {
     stroke(0);
     line(last_x, last_y, mouseX, mouseY);
     last_x = mouseX;
     last_y = mouseY;
     
     stroke(15, 0, 255);
     control_x = viscosity * (mouseX - control_x) + control_x;
     control_y = viscosity * (mouseY - control_y) + control_y;
     line(last_control_x, last_control_y, control_x, control_y);
     last_control_x = control_x;
     last_control_y = control_y;
     
    }
    pointx.append(mouseX);
    pointy.append(mouseY);
    controlx.append(control_x);
    controly.append(control_y);
  }
  else {
   background(255); 
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
  controlx = new FloatList();
  controly = new FloatList();
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
  row.setString("control_x", getString(controlx));
  row.setString("control_y", getString(controly));
  id++;
  
}

void keyPressed() {
  int num = key - '0';
  if (num >= 0 && num < n_styles) {
    style = num;
  }
  else if (key == 's') {
    saveTable(table, str(hour()) + str(minute()) + str(second()) + ".csv");
    println("saved.");
  }
  else if (key == 'd') {
   noSave = !noSave; 
  }
}
