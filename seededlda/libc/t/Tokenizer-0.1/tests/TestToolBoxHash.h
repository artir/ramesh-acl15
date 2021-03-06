/*
** TestToolBoxHash.h
** Login : Julien Lemoine <speedblue@happycoders.org>
** Started on  Fri Jul 14 13:43:14 2006 Julien Lemoine
** $Id$
** 
** Copyright (C) 2006 Julien Lemoine
** This program is free software; you can redistribute it and/or modify
** it under the terms of the GNU Lesser General Public License as published by
** the Free Software Foundation; either version 2 of the License, or
** (at your option) any later version.
** 
** This program is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU Lesser General Public License for more details.
** 
** You should have received a copy of the GNU Lesser General Public License
** along with this program; if not, write to the Free Software
** Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/

#ifndef   	TESTTOOLBOXHASH_H_
# define   	TESTTOOLBOXHASH_H_

#include <cppunit/extensions/HelperMacros.h>

namespace UnitTest
{
  /**
   *
   * @brief Hash test suite
   *
   * <h2>Try to add/get string from hash</h2>
   *
   * @author Julien Lemoine <speedblue@happycoders.org>
   *
   */

  class TestToolBoxHash : public CppUnit::TestCase
    {
	  CPPUNIT_TEST_SUITE(TestToolBoxHash);
	  CPPUNIT_TEST(testString);
	  CPPUNIT_TEST(testUnsigned);
	  CPPUNIT_TEST_SUITE_END();

    public:
	  /// run all test tokenizer
	  void testString();
	  void testUnsigned();
    };
}

#endif 	    /* !TESTTOOLBOXHASH_H_ */
