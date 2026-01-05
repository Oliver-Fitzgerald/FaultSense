#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Widget.H>
#include <FL/Fl_Tabs.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Flex.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Button.H>
#include <iostream>

int main(int argc, char **argv){

    int screen_w = Fl::w();
    int screen_h = Fl::h();

    Fl_Window *sandboxWindow = new Fl_Window(screen_w, screen_h);
    sandboxWindow->color(FL_BLUE);

    Fl_Box *group = new Fl_Box(300,100,1300,200);
    {
        Fl_Label *label = new Fl_Label;
        label->draw(310, 110,300,30, FL_ALIGN_CENTER);
        
    }
    group->color(FL_RED);
    group->box(FL_FLAT_BOX);

    sandboxWindow->end();
    sandboxWindow->show(argc,argv);
    return  Fl::run();
}
